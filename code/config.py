import os
from dotenv import dotenv_values

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# OpenAI official API. The key is read only from the repository's .env (git-ignored),
# never from code or the shell environment.
ENV_PATH = os.path.join(REPO_DIR, '.env')
OPENAI_API_BASE = 'https://api.openai.com/v1'
# The paper's GPT-4 Vision-Preview and GPT-4 Turbo are retired; one model serves the
# vision and text steps (CHECKLIST.md A3).
MODELS = {
    'llm': 'gpt-4o-2024-08-06',
    'llm_vision': 'gpt-4o-2024-08-06',
    'embedding': 'text-embedding-3-large'
}
# paper.txt line 305: temperature = 0 for all API calls (the original code used 0.01-0.3)
TEMPERATURE = 0
DEFAULT_MODEL = 'gpt-4o-2024-08-06'
COMPARISON_MODEL = 'gpt-6.1-sol'
CHAT_MODELS = (DEFAULT_MODEL, COMPARISON_MODEL)
DESCRIPTION_LIMITS = {DEFAULT_MODEL: 200, COMPARISON_MODEL: 4096}


def select_model(model):
    """Select the same model for text and vision; leave embeddings unchanged (F1)."""
    if model not in CHAT_MODELS:
        raise ValueError(f'unsupported experiment model: {model}')
    MODELS.update(llm=model, llm_vision=model)


def temperature_for(model):
    # F1: Sol rejects 0; explicitly send its supported default, recorded as a deviation
    # from paper.txt:303-305. The baseline retains the paper's temperature 0.
    return 1 if model == COMPARISON_MODEL else TEMPERATURE


def completion_limit(model, limit):
    """SDK 1.12 accepts new wire parameters through extra_body (F3).

    Sol's limit covers both reasoning and visible tokens. Its description budget is
    raised after a development response spent all 200 tokens on reasoning (F3).
    """
    if limit is None:
        return {}
    if model == COMPARISON_MODEL:
        return {'extra_body': {'max_completion_tokens': limit}}
    return {'max_tokens': limit}


def description_limit(model):
    return DESCRIPTION_LIMITS.get(model, 200)

# One JSON line per API response, with the requested and the served model (api_log.py)
API_LOG_PATH = os.path.join(REPO_DIR, 'logs', 'openai_api.jsonl')


def load_api_key(env_path=None):
    """Return OPENAI_API_KEY from .env (CHAT_API_KEY is accepted as well).

    Values are read literally: ${...} is not expanded (python-dotenv would fill it in from
    the shell), and a value that still refers to a variable is rejected.
    """
    env_path = env_path or ENV_PATH
    values = dotenv_values(env_path, interpolate=False)
    api_key = values.get('OPENAI_API_KEY') or values.get('CHAT_API_KEY')
    if not api_key:
        raise RuntimeError(f"No OPENAI_API_KEY in {os.path.normpath(env_path)}")
    if '$' in api_key:
        raise RuntimeError(f"The key in {os.path.normpath(env_path)} refers to a variable ($); write the key itself")
    return api_key
