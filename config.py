import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project directory
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Use an explicit model selection; fallback models must be available to your API key.
_CASCADE_CANDIDATES = [
    os.getenv("AGENT_MODEL", "gemini-2.5-flash"),
    *os.getenv("AGENT_FALLBACK_MODELS", "").split(","),
]
_CASCADE_CANDIDATES = [model.strip() for model in _CASCADE_CANDIDATES if model.strip()]

# Preserve exact order while removing duplicates
EFFICIENT_MODEL_ARCHITECTURES = []
for _m in _CASCADE_CANDIDATES:
    if _m and _m not in EFFICIENT_MODEL_ARCHITECTURES:
        EFFICIENT_MODEL_ARCHITECTURES.append(_m)

DEFAULT_MODEL = EFFICIENT_MODEL_ARCHITECTURES[0]
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "5"))

# Rate Limiting & Internet Search Settings
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "2.5"))  # Min seconds between LLM calls to prevent RPM 429
MAX_SEARCH_SNIPPETS = int(os.getenv("MAX_SEARCH_SNIPPETS", "3"))

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    return key.strip()
