import os
from dotenv import load_dotenv

load_dotenv()

def _required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

GOOGLE_GENAI_API_KEY = _required("GOOGLE_GENAI_API_KEY")