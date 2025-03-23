from decouple import config
from dotenv import load_dotenv
import torch

load_dotenv()

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "deepseek-r1-distill-llama-70b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BRAVE_API_KEY = config("BRAVE_API_KEY")
