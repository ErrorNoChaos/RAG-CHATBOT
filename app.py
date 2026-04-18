import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils.embedder import get_embeddings
from utils.retriever import load_vectorstore, get_retriever
from config import LLM_MODEL, GOOGLE_API_KEY, VECTORSTORE_DIR

# ── Custom prompt to reduce hallucination ─────────────────────────────────────
PROMPT_TEMPLATE = """You are a precise assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I don't have enough information to answer this."
Do NOT use prior knowledge or make up facts.

Context:
{context}

Question: {question}

Answer:"""

def build_rag_chain():
    embeddings = get_embeddings()
    
    # Check if vectorstore exists
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError("Run `python ingest.py` first to build the vector index.")
    
    db = load_vectorstore(embeddings)
    retriever = get_retriever(db, search_type="similarity")
    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,          # Lower = more factual, less creative
        convert_system_message_to_human=True,  # Gemini requires this
    )
    
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # ── LCEL chain (modern LangChain style) ───────────────────────────────────
    def format_docs(docs):
        """Merge retrieved chunks into a single context string."""
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

def main():
    print("Building RAG chain...")
    chain, retriever = build_rag_chain()
    print("Ready! Type 'quit' to exit.\n")
    
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        
        # Optional: show retrieved sources
        retrieved = retriever.invoke(query)
        print(f"\n[Retrieved {len(retrieved)} chunks from: "
              f"{set(d.metadata.get('source') for d in retrieved)}]")
        
        answer = chain.invoke(query)
        print(f"\nAssistant: {answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()