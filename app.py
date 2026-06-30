import streamlit as st
from traditional_rag import rag_backend
import time

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="Indian Legal AI (Traditional RAG)",
    page_icon="⚖️",
    layout="wide"
)

st.title("Indian Legal AI Assistant")
st.markdown("Powered by ChromaDB Vector Search and Llama-3.3-70b")

# ─────────────────────────── SESSION STATE ─────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────── CHAT INTERFACE ────────────────────────
if user_query := st.chat_input("Ask a legal question (e.g., about Bohatti Devi or Aadhaar Act)..."):
    
    # Append user question to UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process response
    with st.chat_message("assistant"):
        status_text = st.empty()
        status_text.text("Searching vector database and generating response...")
        
        try:
            # Call the global backend instance
            result = rag_backend.query(user_query)
            
            # Clear status
            status_text.empty()
            
            # Display final answer
            st.markdown(result["answer"])
            
            # Display metrics and metadata
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Execution Time: {result['execution_time_seconds']} seconds")
            
            # Expander for citations and raw evidence
            with st.expander("View Retrieved Evidence & Citations"):
                st.subheader("Citations Found:")
                if result["citations"]:
                    for citation in result["citations"]:
                        st.markdown(f"- {citation}")
                else:
                    st.markdown("No direct citations retrieved.")
                    
                st.subheader("Raw Context Chunks Sent to LLM:")
                for i, snippet in enumerate(result["evidence_snippets"]):
                    st.text_area(f"Chunk {i+1}", snippet, height=150, disabled=True)
                    
            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            
        except Exception as e:
            status_text.empty()
            st.error(f"An error occurred: {str(e)}")