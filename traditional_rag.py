import os
import re
import time
from typing import Dict, List
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
import warnings

# Suppress noisy warnings
warnings.filterwarnings('ignore', message='.*position_ids.*')
load_dotenv()

# ─────────────────────────── CONFIGURATION ───────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ACTS_DB_DIR = "./chroma_db_acts"
CASES_DB_DIR = "./chroma_db_cases"
LLM_MODEL = "llama-3.3-70b-versatile"

# K-values for retrieval (Number of chunks to fetch)
# Increased slightly to ensure correct context is caught after chunking
CASE_K = 15
ACT_K = 13


# ─────────────────────────── INITIALIZATION ───────────────────────────
class LegalRAGBackend:
    def __init__(self):
        print("Initializing Legal RAG Backend...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Verify databases exist before loading
        if not os.path.exists(ACTS_DB_DIR) or not os.path.exists(CASES_DB_DIR):
            raise FileNotFoundError("Chroma databases not found. Please run ingest_data.py first.")
            
        self.acts_db = Chroma(persist_directory=ACTS_DB_DIR, embedding_function=self.embedding_model)
        self.cases_db = Chroma(persist_directory=CASES_DB_DIR, embedding_function=self.embedding_model)
        
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("Backend ready.")

    def _call_groq(self, prompt: str, system: str = "You are an expert Indian legal AI.") -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    match = re.search("try again in ([0-9.]+)s", err)
                    wait = float(match.group(1)) + 2 if match else 60 * (attempt + 1)
                    print(f"[Rate limit] Waiting {wait:.0f}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Groq rate limit: max retries exceeded.")

    # ─────────────────────────── PIPELINE ───────────────────────────────
    def query(self, user_query: str) -> Dict:
        """
        Executes a traditional retrieve-and-generate pipeline.
        Returns a dictionary containing the final answer and the retrieved citations.
        """
        start_time = time.time()
        
        # 1. RETRIEVE
        # Using similarity_search (exact semantic match) instead of MMR
        case_results = self.cases_db.similarity_search(user_query, k=CASE_K)
        act_results = self.acts_db.similarity_search(user_query, k=ACT_K)
        
        retrieved_evidence = []
        citations = set()
        
        for doc in case_results:
            source = doc.metadata.get("case_name", "Unknown Case")
            citations.add(source)
            retrieved_evidence.append(f"Source: {source}\n{doc.page_content}")
            
        for doc in act_results:
            source = doc.metadata.get("act_name", "Unknown Statute")
            citations.add(source)
            retrieved_evidence.append(f"Source: {source}\n{doc.page_content}")

        evidence_text = "\n\n".join(retrieved_evidence)
        
        # 2. GENERATE
        prompt = (
            "You are an expert Indian legal AI. Answer the following legal query using strictly the provided evidence.\n"
            "Cite your sources inline referencing the specific Case Name or Statute.\n"
            "If the provided evidence does not contain sufficient information to answer the query, "
            "state explicitly that the information is missing from the database. Do not hallucinate external law.\n\n"
            f"Query: {user_query}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        
        final_answer = self._call_groq(prompt)
        
        execution_time = round(time.time() - start_time, 2)
        
        return {
            "answer": final_answer,
            "citations": list(citations),
            "evidence_snippets": retrieved_evidence,
            "execution_time_seconds": execution_time
        }

# Instantiate a single global instance to be imported by app.py
rag_backend = LegalRAGBackend()