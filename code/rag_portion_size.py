import argparse
import os
import re
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from chagApp_openai import Vision
from config import MODELS
import numpy as np
import time
import inflect

# Paths of the original script
RESULTS_PATH = '../df_results.csv'
IMAGE_LINK_PATH = '../df_image_link.csv'
OUTPUT_PATH = '../output.csv'
PORTION_PATH = '../ASA24_GPTFoodCodes_portion.csv'

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def has_common_element(list1, list2):
    return list(set(list1).intersection(set(list2)))

def parse_food_codes(food_code_gpt):
    """The eight-digit codes in a GPTFoodCode answer, in order (str of the answer's lines)."""
    if pd.isna(food_code_gpt):
        return []
    return [int(code) for code in re.findall(r'(?<!\d)\d{8}(?!\d)', food_code_gpt)]

def add_food_code_common(df_results, df_image_link):
    """D1: FoodCodeCommon is the labeled code when the predicted codes include it, else empty.

    The release defines has_common_element but not the code that fills FoodCodeCommon; the
    portion step and nutrient_estimate.py read FoodCodeCommon. FC_Description, the
    description the portion prompts use, is that code's description, so rows without
    FoodCodeCommon have none and the portion step skips them.
    """
    descriptions = df_image_link.drop_duplicates('FoodCode').set_index('FoodCode')['FC_Description']
    common = []
    for i in range(len(df_results)):
        codes = has_common_element(parse_food_codes(df_results.loc[i, 'GPTFoodCode']), [df_results.loc[i, 'FoodCode']])
        common.append(float(codes[0]) if codes else np.nan)
    df_results['FoodCodeCommon'] = common
    df_results['FC_Description'] = df_results['FoodCodeCommon'].map(descriptions)
    return df_results

def extract_unit(value):
    return ' '.join(value.split(' ', 1)[1:])

def extract_amount(value):
    return value.split(' ', 1)[0]

def parse_amount(amount):
    """'3/4' -> 0.75 and the mixed numbers of ASA24, '1-1/2' -> 1.5, on which Fraction()
    raised ValueError; NaN when it is not a number."""
    try:
        return float(sum(Fraction(part) for part in amount.split('-')))
    except (ValueError, ZeroDivisionError):
        return np.nan

def singularize(word, p):
    return p.singular_noun(word) or word

def normalize_portion(text):
    """A portion text compared without case, surrounding spaces and quotes, or a final period."""
    return str(text).strip().strip('"\'').rstrip('.').strip().lower()

def replace_abbreviations(descriptions):
    descriptions = descriptions.replace(to_replace=r'\bNFS\b', value='not further specified subcategory assigned', regex=True)
    return descriptions.replace(to_replace=r'\bNS\b', value='not specified subcategory', regex=True)

def create_prompt(food_description, portion_data, type='shot'):
    # E1, the user's decision (b): each prompt ended with 'If you cannot analyze the image, reply with
    # "I can't help to analyze this image." and specify the reason on a new line.', now dropped
    if type == 'grams':
        # C8: no Nutrition5k prompt was released; this is the amount template of Supplementary
        # Fig. 5 with grams as the unit, asked for every food of the dish in one answer, in the
        # "food: value" lines nutrient_estimate_mix.parse_food_items reads. The last line is the
        # user's decision after the second development run, where gpt-4o-2024-08-06 declined the
        # grams of 8 of 10 dishes
        return f"""The image displays the following foods: {food_description}. Please evaluate the portion size presented and specify the quantity: 
how many {portion_data} of each food are visible in the image? Please respond only with one line per food, in the format "food: numerical value", using the food names above. 
If you cannot determine the exact weight, still give your best estimate in grams."""
    if type == 'shot':
        return f"""The image displays the food described as {food_description}. Please assess the portion size shown in the image and 
choose the option that most accurately represents it from the following choices: {portion_data}. Respond with only the selected portion size."""
    else:
        return f"""The image displays the food described as {food_description}. Please evaluate the portion size presented and specify the quantity: 
how many {portion_data} of the food are visible in the image? Please respond only with the numerical value."""

def update_dataframe(df, index, response, column_desc, column_reason):
    response_splits = response.split('\n')
    df[column_desc] = df[column_desc].astype(object)
    df[column_reason] = df[column_reason].astype(object)
    if response_splits[0] == "I can't help to analyze this image.":
        df.loc[index, column_desc] = response_splits[0]
        df.loc[index, column_reason] = response_splits[1]
    else:
        df.loc[index, column_desc] = response

def process_dataframe(df_results, df_image_link, p):
    ls_label_amount = []
    ls_label_unit = []
    ls_portion_shot = []
    for i in range(len(df_results)):
        str_portion = df_results.loc[i, 'Portion']
        str_food_code = df_results.loc[i, 'FoodCodeCommon']
        ls_label_amount.append(parse_amount(extract_amount(str_portion)))
        ls_label_unit.append(singularize(extract_unit(str_portion), p))
        df_image_link_sel = df_image_link[df_image_link['FoodCode'] == str_food_code].copy()
        df_image_link_sel.sort_values(by=['Multiplier'], ascending=False, inplace=True)
        ls_portions = df_image_link_sel['Portion'].tolist()
        ls_portion_shot.append(' ,'.join(ls_portions))
    df_results['LabelAmount'] = ls_label_amount
    df_results['LabelUnit'] = ls_label_unit
    df_results['PortionShot'] = ls_portion_shot
    df_results['FC_Description'] = df_results['FC_Description'].replace(to_replace=r'\bNFS\b', value='not further specified subcategory assigned', regex=True)
    df_results['FC_Description'] = df_results['FC_Description'].replace(to_replace=r'\bNS\b', value='not specified subcategory', regex=True)
    return df_results

def analyze_portions(df, llm, column_desc, column_reason, portion_data, type='shot', output_path=PORTION_PATH,
                     workers=1, description_column='FC_Description'):
    """Ask about every row with a description. Rows answered in an earlier run are skipped;
    with workers > 1 the rows are asked in parallel (each its own request) instead of one
    every 10 seconds."""
    ls_url_str = df['Link'].tolist()
    lock = threading.Lock()

    def analyze(i, url_str):
        food_description = df.loc[i, description_column]
        try:
            prompt = create_prompt(food_description, portion_data[i], type)
            response = llm.chat(prompt, url_str)
            with lock:
                update_dataframe(df, i, response, column_desc, column_reason)
                df.to_csv(output_path, index=False)
            if workers == 1:
                time.sleep(10)
        except Exception as e:
            print(e)

    todo = [(i, url_str) for i, url_str in enumerate(ls_url_str)
            if not pd.isna(df.loc[i, description_column]) and not pd.isna(portion_data[i]) and pd.isna(df.loc[i, column_desc])]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda item: analyze(*item), todo))
    else:
        for i, url_str in todo:
            analyze(i, url_str)

def main(args):
    # Load data
    df_results = pd.read_csv(args.results_path)
    df_image_link = pd.read_csv(args.image_link_path)

    # Initialize inflect engine
    p = inflect.engine()

    # Process dataframe (D1 adds FoodCodeCommon and FC_Description first)
    df_results = add_food_code_common(df_results, df_image_link)
    df_results = process_dataframe(df_results, df_image_link, p)
    df_results.to_csv(args.output_path, index=False)

    # The original read ASA24_GPTFoodCodes_portion.csv, made by hand from output.csv; it is now
    # started from the processed table, and read again when a run resumes.
    if os.path.exists(args.portion_path):
        df_results_portion = pd.read_csv(args.portion_path)
    else:
        df_results_portion = df_results

    # Initialize Vision API
    llm = Vision(MODELS['llm_vision'])

    # Analyze portions
    if 'GPTPortionDescription' not in df_results_portion.columns:
        df_results_portion['GPTPortionDescription'] = [np.nan] * len(df_results_portion)
        df_results_portion['GPTPortionReason'] = [np.nan] * len(df_results_portion)
    analyze_portions(df_results_portion, llm, 'GPTPortionDescription', 'GPTPortionReason', df_results_portion['PortionShot'],
                     output_path=args.portion_path, workers=args.workers)

    if 'GPTPortionAmount' not in df_results_portion.columns:
        df_results_portion['GPTPortionAmount'] = [np.nan] * len(df_results_portion)
        df_results_portion['GPTPortionAmountReason'] = [np.nan] * len(df_results_portion)
    analyze_portions(df_results_portion, llm, 'GPTPortionAmount', 'GPTPortionAmountReason', df_results_portion['LabelUnit'], type='amount',
                     output_path=args.portion_path, workers=args.workers)
    df_results_portion.to_csv(args.portion_path, index=False)

# D3: FNDDS entries that are not an amount one can see
NOT_A_PORTION = ('Quantity not specified', 'Guideline amount')

def prepare_end_to_end(df, df_food_weights, df_nutrition):
    """D3, without reference labels: the first predicted code that is in FNDDS, its FNDDS
    description, and its FNDDS standard portions as the options."""
    fndds = set(df_nutrition['Food code'])
    df['GPTFoodCodeE2E'] = [next((float(c) for c in parse_food_codes(answer) if c in fndds), np.nan)
                            for answer in df['GPTFoodCode']]
    descriptions = df_nutrition.set_index('Food code')['Main food description']
    df['FC_DescriptionE2E'] = replace_abbreviations(df['GPTFoodCodeE2E'].map(descriptions))
    portions = df_food_weights[~df_food_weights['Portion Description'].str.startswith(NOT_A_PORTION)]
    options = portions.sort_values(['Food Code', 'Seq Num']).groupby('Food Code')['Portion Description'].agg(' ,'.join)
    df['PortionShotE2E'] = df['GPTFoodCodeE2E'].map(options)
    return df

def end_to_end_unit(options, answer):
    """The unit of the chosen FNDDS portion ('1 cup' -> 'cup'); NaN when the answer is not an option."""
    if pd.isna(options) or pd.isna(answer):
        return np.nan
    for option in options.split(' ,'):
        if normalize_portion(option) == normalize_portion(answer):
            return option[2:] if option.startswith('1 ') else option
    return np.nan

def main_end_to_end(args):
    """D3: the shot question over the FNDDS portions of the predicted code, then the amount
    question in the chosen portion's unit. Reads and writes the portion table of main()."""
    df = pd.read_csv(args.portion_path)
    if 'GPTFoodCodeE2E' not in df.columns:
        df = prepare_end_to_end(df, pd.read_csv(args.food_weights_path),
                                pd.read_excel(args.nutrition_path, sheet_name='FNDDS Nutrient Values', skiprows=1))
    for column in ('GPTPortionDescriptionE2E', 'GPTPortionReasonE2E', 'GPTPortionAmountE2E', 'GPTPortionAmountReasonE2E'):
        if column not in df.columns:
            df[column] = [np.nan] * len(df)
    llm = Vision(MODELS['llm_vision'])
    analyze_portions(df, llm, 'GPTPortionDescriptionE2E', 'GPTPortionReasonE2E', df['PortionShotE2E'],
                     output_path=args.portion_path, workers=args.workers, description_column='FC_DescriptionE2E')
    df['PortionUnitE2E'] = [end_to_end_unit(o, a) for o, a in zip(df['PortionShotE2E'], df['GPTPortionDescriptionE2E'])]
    analyze_portions(df, llm, 'GPTPortionAmountE2E', 'GPTPortionAmountReasonE2E', df['PortionUnitE2E'], type='amount',
                     output_path=args.portion_path, workers=args.workers, description_column='FC_DescriptionE2E')
    df.to_csv(args.portion_path, index=False)

# a whole code list and the ';' or ',' after it (a ',' before more digits is no separator, so a
# list at the end of the answer is not cut back to the codes before a comma)
CODE_LIST = re.compile(r'\s*\d{8}(?:\s*,\s*\d{8})*(?!\d)\s*(?:;|,(?!\s*\d))')

def food_name(text):
    """A food name without list markers ('- ', '1. ') and emphasis."""
    return re.sub(r'^(?:[-*•]+|\d+[.)])\s*', '', text.strip()).strip('* ')

def food_items(food_code_gpt):
    """R-C8-1: the foods of a Nutrition5k answer "food: code, code; food: code", as (name, codes),
    for the grams question, the counts and the nutrients alike.

    A colon ends a food name, which runs back to the separator after the previous food's value,
    so names keep their own semicolons (171 FNDDS names have one). The value is the code list
    after the colon, or the text up to the next ';'; a food whose value has no 8-digit code
    keeps an empty code list (a failure, not a food less). A colon with nothing after it
    introduces the list (a disclaimer, "here are some possible matches:"), and text after the
    last value is no food, nor is the empty item of ';;' (an answer line that ended with ';',
    joined with '; '). The same name twice is one food with the codes of both.
    """
    if pd.isna(food_code_gpt):
        return []
    parts = str(food_code_gpt).split(':')
    items = {}
    name = parts[0]
    for i, part in enumerate(parts[1:], 1):
        codes = CODE_LIST.match(part)
        if codes:  # "code, code;" then the next name
            value, next_name = part[:codes.end()], part[codes.end():]
        elif ';' in part:
            value, next_name = part.split(';', 1)
        elif i < len(parts) - 1:  # a colon inside the name ("Beef taco filling: beef, cheese, ...")
            name += ':' + part
            continue
        else:
            value, next_name = part, ''
        if value.strip():
            key = food_name(name)
            known = items.setdefault(key.lower(), (key, []))[1]
            known += [code for code in parse_food_codes(value) if code not in known]
        name = next_name.lstrip('; ')  # the ';' of ';;' was left in the next name ('; spinach')
    return list(items.values())

def food_names(food_code_gpt):
    """The food names of a Nutrition5k answer (C8), for the grams question; NaN when no food has a code."""
    items = food_items(food_code_gpt)
    if not any(codes for _, codes in items):
        return np.nan
    return '; '.join(name for name, _ in items)

def main_nutrition5k(args):
    """C8: grams of every recognized food of each dish, one request per dish."""
    df = pd.read_csv(args.portion_path if os.path.exists(args.portion_path) else args.results_path)
    df['GPTFoodNames'] = df['GPTFoodCode'].map(food_names)
    for column in ('GPTAmount', 'GPTAmountReason'):
        if column not in df.columns:
            df[column] = [np.nan] * len(df)
    llm = Vision(MODELS['llm_vision'])
    analyze_portions(df, llm, 'GPTAmount', 'GPTAmountReason', ['grams'] * len(df), type='grams',
                     output_path=args.portion_path, workers=args.workers, description_column='GPTFoodNames')
    df.to_csv(args.portion_path, index=False)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Portion size estimation for the ASA24 images (paper protocol).")
    parser.add_argument("--results_path", default=RESULTS_PATH, help="Food code results (rag_food_code.py).")
    parser.add_argument("--image_link_path", default=IMAGE_LINK_PATH)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    parser.add_argument("--portion_path", default=PORTION_PATH)
    parser.add_argument("--workers", type=int, default=1, help="Rows asked in parallel.")
    parser.add_argument("--dataset", default="asa24", choices=["asa24", "nutrition5k"])
    parser.add_argument("--end_to_end", action="store_true", help="D3: the portion step without reference labels.")
    parser.add_argument("--food_weights_path", default='../FoodWeights.csv')
    parser.add_argument("--nutrition_path", default='../FNDDS/2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx')
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    if args.dataset == 'nutrition5k':
        main_nutrition5k(args)
    elif args.end_to_end:
        main_end_to_end(args)
    else:
        main(args)
