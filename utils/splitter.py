from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Split documents into overlapping chunks.
    
    WHY CHUNKING?
    - LLMs have context limits; you can't feed an entire book
    - Smaller chunks = more precise retrieval
    - Overlap prevents cutting a sentence/concept in half
    
    TRADEOFFS:
    - Smaller chunks (500):  precise retrieval, risk losing context
    - Larger chunks (2000):  rich context, dilutes retrieval precision
    - Overlap (10-20%):      sweet spot for continuity
    
    RecursiveCharacterTextSplitter tries to split on:
    ["\n\n", "\n", " ", ""] in order — preserves paragraph structure.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,  # adds char position to metadata
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks