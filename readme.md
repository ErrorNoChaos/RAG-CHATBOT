RAG Chatbot | LangChain · FAISS · Google Gemini API · Python

Designed and implemented a production-style Retrieval-Augmented Generation (RAG) pipeline from scratch, enabling context-grounded Q&A over custom documents with zero hallucination tolerance
Built a modular ingestion pipeline using LangChain's RecursiveCharacterTextSplitter, Google Gemini embedding-001, and FAISS vector store with local index persistence — reducing re-embedding overhead on repeated runs
Configured semantic retrieval with Maximal Marginal Relevance (MMR) and implemented hybrid search via EnsembleRetriever (FAISS + BM25), improving recall for both semantic and exact-match queries
Engineered prompt templates enforcing context-only answering, with explicit fallback handling to prevent LLM confabulation
Deployed an interactive Streamlit UI with real-time streaming responses, multi-turn conversational memory, and source citation display

Tech Stack: Python · LangChain (LCEL) · FAISS · Google Gemini API · Streamlit · PyPDF · HuggingFace Cross-Encoders