import os
import re
import time
from typing import List, Dict, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq
import warnings

warnings.filterwarnings('ignore', message='.*position_ids.*')
load_dotenv()

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ACTS_DB_DIR = "./chroma_db_acts"
CASES_DB_DIR = "./chroma_db_cases"
LLM_MODEL = "llama-3.3-70b-versatile"
CASE_K = 5
ACT_K = 3
MAX_EVIDENCE_CHARS = 1200

# State Definition
class LegalAgentState(TypedDict):
    user_query: str
    sub_issues: List[str]
    research_plan: List[str]
    retrieved_evidence: List[Dict[str, str]]
    evaluation_status: str
    final_answer: str
    citations: List[str]
    iteration_count: int

# Initialization
def initialize_backend():
    print("Initializing Agentic Backend...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    if not os.path.exists(ACTS_DB_DIR) or not os.path.exists(CASES_DB_DIR):
        raise FileNotFoundError("Chroma databases not found. Please run ingest_data.py first.")
        
    acts_db = Chroma(persist_directory=ACTS_DB_DIR, embedding_function=embedding_model)
    cases_db = Chroma(persist_directory=CASES_DB_DIR, embedding_function=embedding_model)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    return cases_db, acts_db, client

cases_db, acts_db, client = initialize_backend()

# LLM Helper
def call_groq(prompt: str, system: str = "You are an expert Indian legal AI.") -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                match = re.search("try again in ([0-9.]+)s", err)
                wait  = float(match.group(1)) + 2 if match else 60 * (attempt + 1)
                print(f"[Rate limit] Waiting {wait:.0f}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq rate limit: max retries exceeded.")

# Nodes
def decompose_issue(state: LegalAgentState):
    query = state.get("user_query", "")
    prompt = (
        "Break down the following legal query into 2 to 3 core legal concepts or sub-issues.\n"
        "Output exactly one sub-issue per line. Do not use numbers, bullets, or extra commentary.\n"
        f"Query: {query}"
    )
    
    response = call_groq(prompt).strip()
    sub_issues = [line.strip() for line in response.splitlines() if line.strip()]
    
    return {"sub_issues": sub_issues, "iteration_count": 0, "retrieved_evidence": []}

def plan_research(state: LegalAgentState):
    original_query = state.get("user_query", "")
    sub_issues = state.get("sub_issues", [])
    issues_text = ", ".join(sub_issues)
    
    prompt = (
        "You are formulating search queries for a legal vector database.\n"
        "Generate 3 to 4 specific search queries using the original query and the sub-issues below.\n"
        "CRITICAL INSTRUCTION: If the Original Query mentions a specific Case Name, Statute, or Person, "
        "you MUST include that exact proper noun in your search queries to ensure exact-match retrieval.\n\n"
        f"Original Query: {original_query}\n"
        f"Sub-issues: {issues_text}\n\n"
        "Output exactly one search query per line. Do not use numbers, bullets, or extra commentary."
    )
    
    response = call_groq(prompt).strip()
    plan = [line.strip() for line in response.splitlines() if line.strip()]
    
    return {"research_plan": plan}

def retrieve_evidence(state: LegalAgentState):
    plan = state.get("research_plan", [])
    evidence = state.get("retrieved_evidence", [])
    
    new_evidence = []
    existing_contents = {e["content"] for e in evidence}
    
    for query in plan:
        cases = cases_db.similarity_search(query, k=CASE_K)
        for c in cases:
            content = c.page_content[:MAX_EVIDENCE_CHARS]
            if content not in existing_contents:
                new_evidence.append({
                    "source": c.metadata.get("case_name", "Case Law"),
                    "content": content
                })
                existing_contents.add(content)
        
        acts = acts_db.similarity_search(query, k=ACT_K)
        for a in acts:
            content = a.page_content[:MAX_EVIDENCE_CHARS]
            if content not in existing_contents:
                new_evidence.append({
                    "source": a.metadata.get("act_name", "Statute"),
                    "content": content
                })
                existing_contents.add(content)
                
    return {"retrieved_evidence": evidence + new_evidence}

def evaluate_and_refine(state: LegalAgentState):
    query = state.get("user_query", "")
    evidence = state.get("retrieved_evidence", [])
    iteration = state.get("iteration_count", 0) + 1
    
    max_iterations = 3
    if iteration >= max_iterations:
        return {"evaluation_status": "complete", "iteration_count": iteration}
        
    context = "\n".join([f"[{e['source']}] {e['content'][:300]}..." for e in evidence])
    
    prompt = (
        f"User Query: {query}\n\n"
        f"Current Evidence Snippets:\n{context}\n\n"
        "Does the current evidence provide sufficient grounds to thoroughly answer the legal query? "
        "Reply strictly with YES or NO on the first line.\n"
        "If NO, provide exactly one new, highly specific search phrase to find the missing legal context on the second line."
    )
    
    response = call_groq(prompt).strip()
    lines = response.splitlines()
    decision = lines[0].strip().upper()
    
    if "YES" in decision:
        return {"evaluation_status": "complete", "iteration_count": iteration}
    else:
        new_issue = lines[1].strip() if len(lines) > 1 else f"further context for {query}"
        return {"evaluation_status": "incomplete", "iteration_count": iteration, "sub_issues": [new_issue]}

def generate_reasoning(state: LegalAgentState):
    query = state.get("user_query", "")
    evidence = state.get("retrieved_evidence", [])
    
    evidence_text = "\n\n".join([f"Source: {e['source']}\n{e['content']}" for e in evidence])
    
    prompt = (
        "You are an expert Indian legal AI. Answer the following legal query using strictly the provided evidence.\n"
        "Cite your sources inline referencing the specific Case Name or Statute.\n"
        "If the evidence does not contain the answer, state that explicitly. Do not hallucinate external law.\n\n"
        f"Query: {query}\n\n"
        f"Evidence:\n{evidence_text}"
    )
    
    final_answer = call_groq(prompt)
    citations = list(set([e['source'] for e in evidence]))
    
    return {"final_answer": final_answer, "citations": citations}

def router(state: LegalAgentState):
    status = state.get("evaluation_status", "incomplete")
    if status == "complete":
        return "done"
    return "loop"

# Graph Construction
def build_graph():
    workflow = StateGraph(LegalAgentState)

    workflow.add_node("decomposer", decompose_issue)
    workflow.add_node("planner",    plan_research)
    workflow.add_node("retriever",  retrieve_evidence)
    workflow.add_node("evaluator",  evaluate_and_refine)
    workflow.add_node("generator",  generate_reasoning)

    workflow.add_edge(START,        "decomposer")
    workflow.add_edge("decomposer", "planner")
    workflow.add_edge("planner",    "retriever")
    workflow.add_edge("retriever",  "evaluator")

    workflow.add_conditional_edges(
        "evaluator",
        router,
        {
            "loop": "planner",
            "done": "generator"
        }
    )

    workflow.add_edge("generator", END)
    return workflow.compile()

agentic_app = build_graph()