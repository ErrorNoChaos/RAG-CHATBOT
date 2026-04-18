import os
from dotenv import load_dotenv

load_dotenv()

# Model config
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-flash-latest"       # Use "gemini-1.5-pro" for higher quality

# Chunking
CHUNK_SIZE = 1500      # Characters per chunk
CHUNK_OVERLAP = 150    # Overlap prevents context loss at boundaries

# Retrieval
RETRIEVAL_K = 4        # Number of chunks to retrieve per query

# Paths
DATA_DIR = "data/"
VECTORSTORE_DIR = "vectorstore/"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY, "GOOGLE_API_KEY not set in .env"