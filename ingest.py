from utils.loader import load_documents
from utils.splitter import split_documents
from utils.embedder import get_embeddings
from utils.retriever import build_vectorstore
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def ingest():
    print("=== Starting ingestion ===")
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = get_embeddings()
    build_vectorstore(chunks, embeddings)
    print("=== Ingestion complete ===")

if __name__ == "__main__":
    ingest()