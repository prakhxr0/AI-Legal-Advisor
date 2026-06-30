import streamlit as st
from agentic_rag_core import agentic_app
import time

st.set_page_config(
    page_title="Agentic Legal AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("Agentic Indian Legal AI")
st.markdown("Powered by LangGraph and Llama-3.3-70b")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a complex legal question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status = st.empty()
        status.info("Agent is decomposing the query and planning research...")
        
        start_time = time.time()

        # Initialize the LangGraph state
        initial_state = {
            "user_query": user_query,
            "sub_issues": [],
            "research_plan": [],
            "retrieved_evidence": [],
            "evaluation_status": "",
            "final_answer": "",
            "citations": [],
            "iteration_count": 0
        }

        try:
            # Invoke the LangGraph workflow
            result = agentic_app.invoke(initial_state)
            
            status.empty()
            
            # Display final generated answer
            st.markdown(result["final_answer"])
            
            execution_time = round(time.time() - start_time, 2)
            st.caption(f"Execution Time: {execution_time} seconds")

            # Display Agentic Reasoning Trace
            with st.expander("Inspect Agentic Reasoning Trace"):
                st.subheader("Sub-Issues Identified")
                for issue in result["sub_issues"]:
                    st.write(f"- {issue}")
                    
                st.subheader("Vector Search Queries Executed")
                for query in result["research_plan"]:
                    st.write(f"🔍 {query}")
                    
                st.subheader("Citations Found")
                if result["citations"]:
                    for citation in result["citations"]:
                        st.write(f"📄 {citation}")
                else:
                    st.write("No citations extracted.")
                    
                st.subheader("Evaluation Metric")
                st.write(f"Total Iteration Loops: {result['iteration_count']} / 3")
                
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["final_answer"]
            })
            
        except Exception as e:
            status.empty()
            st.error(f"An error occurred within the agent graph: {str(e)}")