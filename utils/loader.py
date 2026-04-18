from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
import os

def load_documents(data_dir: str) -> list[Document]:
    """
    Load all TXT and PDF files from a directory.
    Returns a flat list of LangChain Document objects.
    Each Document has .page_content (str) and .metadata (dict).
    """
    docs = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        
        if fname.endswith(".txt"):
            loader = TextLoader(fpath, encoding="utf-8")
        elif fname.endswith(".pdf"):
            loader = PyPDFLoader(fpath)  # splits by page automatically
        else:
            continue  # skip unsupported formats
        
        loaded = loader.load()
        # Attach source metadata — useful for filtering later
        for doc in loaded:
            doc.metadata["source"] = fname
        docs.extend(loaded)
    
    print(f"Loaded {len(docs)} document chunks from {data_dir}")
    return docs