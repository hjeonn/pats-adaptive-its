import os
import torch

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") 
HF_TOKEN = os.getenv("HF_TOKEN", "")       
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")   

MODEL_TYPES = ["gpt-3.5-turbo", "gpt-4", "gemini-2.5-flash", "t5-large", "llama-3.2-1b", "mistral-7b"]
PROMPT_TYPES = ["wo-pats", "pats"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CIMA_FILE_PATH = "cima_dataset.json"
TSCC_FILE_PATH = "tscc_processed_conversations.json"
DATA_NAME = "TSCC"
OUTPUT_DIR = "results"