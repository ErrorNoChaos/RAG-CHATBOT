from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import EMBEDDING_MODEL, GOOGLE_API_KEY

def get_embeddings():
    """
    Embeddings convert text → dense vectors (768 dimensions for embedding-001).
    Semantically similar text → similar vectors → close in vector space.
    This is what makes semantic search possible.
    
    CRITICAL: Use the SAME embedding model at ingestion AND query time.
    Mixing models corrupts retrieval.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"  # optimize for retrieval (not classification)
    )