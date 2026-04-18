import os
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import VECTORSTORE_DIR, RETRIEVAL_K

def build_vectorstore(chunks: list[Document], embeddings) -> FAISS:
    """
    FAISS (Facebook AI Similarity Search) builds an index over embedding vectors.
    At query time, it finds the k-nearest vectors using L2 or cosine distance.
    It's in-memory and lightning fast — ideal for < 1M chunks.
    """
    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    db.save_local(VECTORSTORE_DIR)  # saves index.faiss + index.pkl
    print(f"FAISS index saved to {VECTORSTORE_DIR}")
    return db

def load_vectorstore(embeddings) -> FAISS:
    """Load pre-built index from disk. Much faster than rebuilding."""
    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True  # required flag in recent LangChain
    )

def get_retriever(db: FAISS, search_type: str = "similarity"):
    """
    search_type options:
    - "similarity": pure cosine/L2 similarity (fast, default)
    - "mmr": Maximal Marginal Relevance — balances relevance + diversity
             Use MMR when you notice retrieved chunks are all near-duplicates.
    
    k=4 is a good default. Increase if answers miss details; 
    decrease if the LLM gets confused by too much context.
    """
    if search_type == "mmr":
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVAL_K, "fetch_k": 20, "lambda_mult": 0.7}
        )
    return db.as_retriever(search_kwargs={"k": RETRIEVAL_K})