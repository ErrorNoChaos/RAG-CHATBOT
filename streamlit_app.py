import streamlit as st
from app import build_rag_chain

st.title("RAG Chatbot")
st.caption("Powered by Gemini + FAISS")

@st.cache_resource
def get_chain():
    return build_rag_chain()

chain, retriever = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Streaming
            response = st.write_stream(chain.stream(prompt))
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Show sources in expander
    docs = retriever.invoke(prompt)
    with st.expander("Sources"):
        for d in docs:
            st.caption(f"**{d.metadata.get('source')}** — {d.page_content[:200]}...")