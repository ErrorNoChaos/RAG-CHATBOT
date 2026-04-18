RAG Chatbot with Gemini & FAISS

  A professional Retrieval-Augmented Generation (RAG) chatbot that allows users
  to chat with their own documents. This project implements a full pipeline:
  from document ingestion and vectorization to a conversational AI interface.

  🚀 Overview

  This chatbot uses Google Gemini as the LLM and FAISS (Facebook AI Similarity
  Search) as the vector store. Instead of relying on the LLM's general
  knowledge, the system retrieves the most relevant chunks of text from uploaded
   documents to provide precise, fact-based answers.

  🛠️ Tech Stack

  - LLM: Google Gemini (langchain-google-genai)
  - Orchestration: LangChain (LCEL - LangChain Expression Language)
  - Vector Store: FAISS
  - Frontend: Streamlit
  - Embeddings: Google Generative AI Embeddings

  📂 Project Architecture

  1. Ingestion Pipeline (ingest.py)

  The ingestion process converts raw documents into a searchable mathematical
  format (embeddings):
  - Loading: Reads files from the data/ directory.
  - Splitting: Breaks large documents into smaller, manageable chunks (defined
  by CHUNK_SIZE and CHUNK_OVERLAP in config.py) to ensure the LLM receives the
  most relevant context without hitting token limits.
  - Embedding: Converts text chunks into high-dimensional vectors.
  - Indexing: Stores these vectors in a FAISS index for lightning-fast
  similarity searches.

  2. The RAG Chain (app.py)

  The core logic uses a modern LCEL chain:
  - Retriever: When a user asks a question, the system searches the FAISS index
  for the top-k most similar text chunks.
  - Contextual Prompting: The retrieved chunks are injected into a strict system
   prompt that forces the AI to answer only using the provided context,
  significantly reducing hallucinations.
  - Generation: Gemini processes the context and the question to generate a
  natural language response.

  3. User Interface (streamlit_app.py)

  A clean, interactive chat interface built with Streamlit:
  - Session State: Maintains chat history across interactions.
  - Streaming: Uses st.write_stream for a "typing" effect as the AI responds.
  - Transparency: Includes a "Sources" expander that shows exactly which parts
  of the documents were used to generate the answer.

  ⚙️ Setup & Installation

  Prerequisites

  - Python 3.11+
  - A Google AI Studio API Key

  Installation

  1. Clone the repo:
  git clone <your-repo-url>
  cd RAG-chatbot
  2. Setup Environment:
  python -m venv venv
  source venv/bin/activate  # Mac/Linux
  # OR .\venv\Scripts\activate # Windows
  pip install -r requirements.txt
  3. Configuration:
  Create a .env file or update config.py with your API key:
  GOOGLE_API_KEY = "your_api_key_here"

  🏃 How to Use

  Step 1: Add Data

  Place your PDFs or text files in the data/ folder.

  Step 2: Ingest Data

  Run the ingestion script to build the vector database:
  python ingest.py

  Step 3: Launch the App

  Start the Streamlit interface:
  streamlit run streamlit_app.py

  🛡️ Key Features

  - Anti-Hallucination: Strict prompt engineering ensures the AI admits when it
  doesn't have enough information.
  - High Performance: FAISS allows for near-instant retrieval even with large
  document sets.
  - Source Attribution: Every answer is backed by a source citation from the
  original documents.
