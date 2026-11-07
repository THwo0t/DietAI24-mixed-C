"""Stage model split; only ASA portion changes in this frozen comparison."""
MODELS = {"llm": "gpt-6.1-sol", "llm_vision": "gemini-3.8-flash", "embedding": "text-embedding-3-large"}
STAGE_MODELS = {"recognition": "gemini-3.8-flash", "selection": "gemini-3.8-flash",
                "portion": {"nutrition5k": "gemini-3.8-flash", "asa24": "gpt-6.1-sol"}}
CHAT_BASE_URL = "https://api.zicc.cc/v1"
EMBEDDING_BASE_URL = "https://api.openai.com/v1"
