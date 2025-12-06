"""
Configuration settings for the Multimodal Learning Coach Agent
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class Config:
    """Configuration settings"""
    device = 'cpu'
    # Model path - adjust this to your model location
    MODEL_PATH = str(SRC_DIR / "Qwen3-VL-2B-Instruct")  # Use non-FP8 version
    MODEL_NAME = "Qwen3-VL-2B-Instruct"

    # Generation settings
    MAX_TOKENS = 512  # Increased from 128 for complete explanations
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 3  # For lesson retrieval

    # Advanced generation settings
    DO_SAMPLE = True
    REPETITION_PENALTY = 1.1  # Reduce repetition
    LENGTH_PENALTY = 1.0  # Neutral length penalty

    # Image processing
    MIN_PIXELS = 224 * 224
    MAX_PIXELS = 1280 * 1280

    # Paths
    KNOWLEDGE_BASE_PATH = str(DATA_DIR / "knowledge_base.json")
    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight sentence transformer
