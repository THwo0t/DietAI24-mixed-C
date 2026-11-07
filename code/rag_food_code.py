import argparse
import warnings
import os
import re
import logging
import threading
import requests
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter

from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb.config
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from config import MODELS, OPENAI_API_BASE, temperature_for, completion_limit, description_limit, load_api_key
from api_log import openai_client
from chagApp_openai import image_url
from rag_portion_size import food_items, parse_food_codes

# C1: the FNDDS index is built once and kept on disk (paper.txt:158-159)
INDEX_DIR = '../index/chroma'


def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

def configure_chat_openai(model_key, MODELS, api_key):
    """Create a ChatOpenAI configuration based on the specified model key."""
    limits = completion_limit(MODELS[model_key], description_limit(MODELS[model_key]) if model_key == 'llm_vision' else None)
    if 'extra_body' in limits:
        limits = {'model_kwargs': limits}  # old LangChain passes extra_body to SDK 1.12
    return ChatOpenAI(
        model=MODELS[model_key],
        openai_api_key=api_key,
        openai_api_base=OPENAI_API_BASE,
        client=openai_client().chat.completions,
        temperature=temperature_for(MODELS[model_key]),
        **limits
    )

def initialize_clients(MODELS):
    api_key = load_api_key()
    llm = configure_chat_openai('llm', MODELS, api_key)
    llm_vision = configure_chat_openai('llm_vision', MODELS, api_key)
    embedding = OpenAIEmbeddings(
        model=MODELS['embedding'],
        openai_api_key=api_key,
        openai_api_base=OPENAI_API_BASE,
        client=openai_client().embeddings
    )
    return llm, llm_vision, embedding

def load_data(file_path):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    return data

def setup_vector_database(data, embedding, persist_directory=INDEX_DIR):
    # Built once and kept on disk; ids are the CSV rows, so a retry after a partial build
    # adds only what is missing. Chroma's anonymous telemetry is off.
    settings = chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True)
    vectordb = Chroma(collection_name="openai_embed", embedding_function=embedding,
                      persist_directory=persist_directory, client_settings=settings)
    if vectordb._collection.count() == len(data):
        logging.info(f"Loaded the vector database from {persist_directory}")
        return vectordb
    ids = [str(doc.metadata['row']) for doc in data]
    attempts = 0
    max_attempts = 5  # You can set this to the number of retry attempts you find appropriate
    while attempts < max_attempts:
        try:
            stored = set(vectordb.get(include=[])['ids'])
            missing = [(doc, i) for doc, i in zip(data, ids) if i not in stored]
            batch = vectordb._client.max_batch_size  # Chroma rejects larger additions (from_documents batched too)
            for start in range(0, len(missing), batch):
                vectordb.add_documents([doc for doc, _ in missing[start:start + batch]],
                                       ids=[i for _, i in missing[start:start + batch]])
            return vectordb  # Successful creation, return the database
        except Exception as e:
            logging.error(f"Failed to setup vector database: {e}")
            attempts += 1
            time.sleep(180)  # Wait for 30 seconds before retrying
            
    # If the loop completes without returning, log final failure and optionally raise an exception
    logging.critical("Failed to initialize vector database after multiple attempts.")
    raise Exception("Failed to initialize vector database after multiple attempts.")

# Helper functions for message generation and checkpointing
def get_messages_from_url(url_str):
    """Generate a sequence of messages for a URL containing an image."""
    return [
        SystemMessage(
            content="You are an expert at analyzing images with computer vision. "
                    "I will present you with a picture of food, which might be placed on a plate, inside a spoon, "
                    "or contained within different vessels. Your job is to accurately identify the food depicted in the image."
        ),
        HumanMessage(
            content=[
                # E1, the user's decision (b): the published last sentence, "If you can't provide assistance
                # with this image, simply respond with 'I can't help to analyze this image.' and provide the
                # reason on a new line.", is dropped; gpt-4o-2024-08-06 took it for most images
                {"type": "text", "text": """Could you name the food shown in the image? If feasible, specify its variety
                 based on ingredients and preparation methods. Ensure your response is brief and avoid
                 speculating on uncertain details."""},
                {"type": "image_url", "image_url": {"url": image_url(url_str)}}
            ]
        )
    ]

def setup_retrieval_prompt():
    RETRIEVE_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given food descriptions to retrieve relevant food information from a vector 
        database. These decriptions should docus purely on the qualitative asepcts of the food, including its broader category, specific flavor profiles, ingredients, and methods of preparation. 
        By generating multiple perspectives on the food description, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Please provide these alternate food descriptions, adding the original description as the sixth entry without mentioning that it is the original description. 
        Ensure each of the six descriptions is separated by a newline.
        Original food description: {question}""",
    )
    return RETRIEVE_PROMPT

def configure_retrievers(llm, vectordb, prompt):
    # llm_chain_retriever = LLMChain(llm=llm, prompt=prompt)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), 
        llm=llm, 
        prompt=prompt
    )
    return retriever_from_llm

# C8: the release has no food code prompt for mixed dishes. nutrient_estimate_mix.py reads the
# answer as "food: code, code; food: code", so for Nutrition5k this sentence asks for that.
CODE_FORMAT = {
    'asa24': "Please provide only the eight-digit codes without any extra information.",
    'nutrition5k': "The image may show several foods. For each food, give its name followed by its eight-digit codes, "
                   "in the format 'food name: code, code; food name: code', without any extra information.",
}

def setup_code_prompt_chain(retriever_from_llm, llm, dataset='asa24'):
    # C4: the vision model answers from the image and the candidates (paper.txt:173-178,
    # Supplementary Fig. 5); the image is attached instead of its URL being written in the text.
    template = """Could you identify and provide the eight-digit food code corresponding to the given food image, 
    considering only the context provided: {context}?

    """ + CODE_FORMAT[dataset] + """ If an exact match for the food description is not available, please identify the closest equivalent. 
    Should there be no relevant food codes based on the context provided, simply reply with 'No appropriate food codes found from the context information.
    """
    CODE_PROMPT = PromptTemplate.from_template(template)

    def code_messages(inputs):
        return [HumanMessage(content=[
            {"type": "text", "text": CODE_PROMPT.format(context=inputs["context"])},
            {"type": "image_url", "image_url": {"url": image_url(inputs["url_str"])}}
        ])]

    # C3: the retriever gets the description only (it got the whole input dict, image URL
    # included); the output keeps the retrieved candidates ("context") next to the "answer".
    food_code_chain = (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever_from_llm)
        | RunnablePassthrough.assign(answer=RunnableLambda(code_messages) | llm | StrOutputParser())
    )
    return food_code_chain

def candidate_codes(documents):
    """Food codes of the retrieved FNDDS rows (CSVLoader writes 'Food code: ...' first)."""
    return [re.match(r'Food code: (\d+)', doc.page_content).group(1) for doc in documents]

def code_model(llm_vision):
    """The vision model for the food code step, on the same client, without the 200-token cap
    of the description step: the text model it replaces had none."""
    return ChatOpenAI(model=llm_vision.model_name, openai_api_key=llm_vision.openai_api_key,
                      openai_api_base=llm_vision.openai_api_base, client=llm_vision.client,
                      temperature=temperature_for(llm_vision.model_name))

def is_integer(s):
    """Check if a string can be converted to an integer."""
    try:
        int(s)
        return True
    except ValueError:
        return False

def has_food_code(food_code_gpt, dataset='asa24'):
    """The stored answer has a food code, so later passes skip the image."""
    if pd.isna(food_code_gpt):
        return False
    if dataset == 'nutrition5k':  # "food: code, ..." starts with a name; a food with a code (R-C8-1)
        return any(codes for _, codes in food_items(food_code_gpt))
    # R-C9-1: an 8-digit code as D1, D4 and D6 read it; the original tested whether characters
    # 3-10 were an integer, which asked "['Food code: 63101000']" again and skipped "['631010001']"
    return len(parse_food_codes(food_code_gpt)) > 0

def image_accessible(url):
    """URLs are checked with a HEAD request, local images (B6) on disk."""
    if url.startswith(('http://', 'https://')):
        return requests.head(url, timeout=5).status_code == 200
    return os.path.isfile(url)

def process_image_url(index, url, llm_vision, df, food_code_chain, results_path, lock=None, dataset='asa24'):
    """Process a single image URL and update the DataFrame with food descriptions and codes.

    GPTAttempts counts the passes that processed the image (C9); with parallel images, the
    lock guards the table and its CSV.
    """
    logging.info(f"Processing URL at index {index}: {url}")
    lock = lock or threading.Lock()
    with lock:
        attempts = df.loc[index, 'GPTAttempts'] if 'GPTAttempts' in df.columns else np.nan
        df.loc[index, 'GPTAttempts'] = 1 if pd.isna(attempts) else attempts + 1
    candidates = np.nan
    try:
        if not image_accessible(url):
            raise Exception("Image URL is not accessible")
        
        IMAGE_PROMPT = get_messages_from_url(url)
        attempts = 0  # Counter for retry attempts
        max_attempts = 5 # Set the maximum number of retry attempts
        
        while True:
            try:
                image_response = llm_vision.invoke(IMAGE_PROMPT)
                food_description = image_response.content
                food_description_splits = food_description.split('\n')
                if food_description_splits[0] == "I can't help to analyze this image.":
                    raise Exception(food_description_splits[-1])
                break
            except Exception as e:
                if '429' in str(e) or 'rate limit' in str(e).lower():  # Check if the exception is due to rate limiting
                    if attempts < max_attempts - 1:
                        sleep_time = min(2 ** attempts * 30, 240)  # Exponential backoff with a maximum wait time
                        logging.warning(f"Rate limit exceeded. Retrying in {sleep_time} seconds.")
                        attempts += 1
                        time.sleep(sleep_time)
                        continue  # Continue to retry after sleeping
                    else:
                        logging.error("Maximum retry attempts reached. Failing with error.")
                        raise  # Exception to indicate max attempts have been reached
                else:
                    logging.error(f"Failed to invoke llm_vision at index {index}: {e}")
                    raise  # Re-raise the exception if it's not related to rate limits
                    
        food_code_result = food_code_chain.invoke({"question": food_description, "url_str": url})
        food_code_response = food_code_result["answer"]
        candidates = str(candidate_codes(food_code_result["context"]))
        if food_code_response == "No appropriate food codes found from the context information.":
            raise Exception(food_code_response)

        if dataset == 'nutrition5k':  # one "food: codes" item per line is read the same way
            food_code = '; '.join(line.strip() for line in food_code_response.split('\n') if line.strip())
        else:
            food_code = str(food_code_response.split('\n'))
        with lock:
            df.loc[index, 'GPTFoodDescription'] = food_description
            df.loc[index, 'GPTFoodCode'] = food_code
    except Exception as e:
        with lock:
            df.loc[index, 'GPTFoodDescription'] = str(e)
            df.loc[index, 'GPTFoodCode'] = np.nan
        # logging.error(f"Failed to process URL at index {index}: {e}")
    finally:
        with lock:
            df.loc[index, 'GPTCandidateCodes'] = candidates
            df.to_csv(results_path, index=False)
        logging.info(f"Data saved to {results_path}")

def load_checkpoint(checkpoint_path):
    """Load the last processed indices from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as file:
            content = file.read()
            if content:
                num, index = content.split(',')
                return int(num), int(index)
    return 0, 0

def save_checkpoint(num, index, checkpoint_path):
    """Save the current indices to a checkpoint file."""
    with open(checkpoint_path, 'w') as file:
        file.write(f"{num},{index}")
    logging.info(f"Checkpoint saved to {checkpoint_path} at num {num}, index {index}")

def process_image_urls(results_path, checkpoint_path, llm_vision, food_code_chain, workers=1, dataset='asa24'):
    """Process image URLs with checkpointing that includes iterations and index.

    With workers > 1 the images of a pass are processed in parallel, each still with its own
    requests; the checkpoint is then saved after each pass.
    """
    num_iterations = 5
    df_url = pd.read_csv(results_path, dtype={'GPTFoodDescription': object, 'GPTFoodCode': object,
                                              'GPTCandidateCodes': object})
    if 'GPTCandidateCodes' not in df_url.columns:
        df_url['GPTCandidateCodes'] = pd.Series(dtype=object)
    last_num, last_index = load_checkpoint(checkpoint_path)
    lock = threading.Lock()

    for num in range(last_num, num_iterations):
        start_index = last_index if num == last_num else 0
        pending = []
        for i in range(start_index, len(df_url)):
            if has_food_code(df_url.loc[i, 'GPTFoodCode'], dataset):
                continue
            pending.append(i)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda i: process_image_url(i, df_url.loc[i, 'Link'], llm_vision, df_url,
                                                          food_code_chain, results_path, lock, dataset), pending))
            save_checkpoint(num + 1, 0, checkpoint_path)
        else:
            for i in pending:
                url = df_url.loc[i, 'Link']
                process_image_url(i, url, llm_vision, df_url, food_code_chain, results_path, lock, dataset)
                save_checkpoint(num, i + 1, checkpoint_path)  # Update checkpoint after each URL
            
        last_index = 0  # Reset last_index after completing each num iteration

def main(args):
    setup_logging(args.log_path)
    logging.info("Starting image processing script")
    llm, llm_vision, embedding = initialize_clients(MODELS)
    data = load_data(args.csv_file)
    vectordb = setup_vector_database(data, embedding, args.index_dir)  # Ensure 'embedding' is initialized

    retrieval_prompt = setup_retrieval_prompt()
    retriever_from_llm = configure_retrievers(llm, vectordb, retrieval_prompt)
    food_code_chain = setup_code_prompt_chain(retriever_from_llm, code_model(llm_vision), args.dataset)
    # Process image URLs (assuming function `process_image_urls` is properly defined)
    process_image_urls(args.results_file, args.checkpoint_file, llm_vision, food_code_chain, args.workers, args.dataset)
    
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    parser = argparse.ArgumentParser(description="Process image URLs and update with food descriptions.")
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing data.")
    parser.add_argument("--checkpoint_file", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--results_file", required=True, help="Path to the results file containing image URLs.")
    parser.add_argument("--log_path", default="process_images.log", help="Path to the log file.")
    parser.add_argument("--index_dir", default=INDEX_DIR, help="Directory of the persisted vector database.")
    parser.add_argument("--workers", type=int, default=1, help="Images processed in parallel within a pass.")
    parser.add_argument("--dataset", default="asa24", choices=sorted(CODE_FORMAT), help="Answer format of the food code step.")
    
    args = parser.parse_args()
    main(args)
