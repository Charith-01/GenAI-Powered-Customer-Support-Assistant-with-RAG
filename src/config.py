import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


# ---------------------------------------------------------
# Load .env
# ---------------------------------------------------------

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)


# ---------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)


# ---------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------

def validate_config():
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to the project root .env file."
        )

    if not GEMINI_MODEL:
        raise ValueError(
            "GEMINI_MODEL is missing."
        )