import os
import re

# Ensure PyThaiNLP uses local folder inside the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHAINLP_DIR = os.path.join(BASE_DIR, ".pythainlp-data")
os.makedirs(PYTHAINLP_DIR, exist_ok=True)
os.environ["PYTHAINLP_DATA_DIR"] = PYTHAINLP_DIR

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

THAI_STOPWORDS = set(thai_stopwords())

def clean_text(text: str) -> str:
    """
    Cleans Thai review text:
    - Removes URLs
    - Normalizes multiple spaces
    - Removes non-Thai/English/number/basic punctuation characters
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove unwanted special characters, keeping Thai, English, digits, spaces
    text = re.sub(r'[^\w\sก-๙]', ' ', text)
    # Normalize multiple whitespace characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_thai(text: str, remove_stop: bool = False) -> list:
    """
    Tokenizes Thai text using PyThaiNLP dictionary-based newmm engine.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    tokens = word_tokenize(cleaned, engine="newmm", keep_whitespace=False)
    if remove_stop:
        tokens = [t for t in tokens if t not in THAI_STOPWORDS and len(t) > 1]
    return tokens

def preprocess_for_display(text: str) -> dict:
    """
    Returns step-by-step preprocessing artifacts for explainability in UI.
    """
    cleaned = clean_text(text)
    tokens = tokenize_thai(cleaned, remove_stop=False)
    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "token_count": len(tokens),
        "tokenized_display": " | ".join(tokens)
    }

