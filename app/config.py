
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGEBASE_DIR = PROJECT_ROOT / "knowledgebase"
DATA_DIR = KNOWLEDGEBASE_DIR / "data"
FAISS_DIR = KNOWLEDGEBASE_DIR / "faiss_index"



# Model config
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.2

# Agent config
MAX_RETRIES = 2
