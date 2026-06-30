# Agentic Indian Legal AI

This repository implements a high-precision legal AI system designed for the Indian legal domain. It utilizes a hybrid architecture that combines an **Agentic RAG pipeline** for complex reasoning with **QAT-optimized fine-tuning** for domain-specific comprehension.

### Core Technical Architecture

*   **Agentic RAG Pipeline**: Built using LangGraph, the system executes multi-step query decomposition and conditional refinement loops. Instead of a single retrieval step, the agent decomposes legal queries, executes targeted searches, evaluates sufficiency, and iterates until the evidence context is sufficient to answer the query accurately.
*   **Domain-Specific Optimization**: The backbone includes a Gemma4-E2B QAT-fine-tuned model. We employed Parameter-Efficient Fine-Tuning (PEFT) and QLoRA to align model performance with complex Indian legal datasets while maintaining computational efficiency.
*   **Evaluation Metrics**: The pipeline achieves a **0.889 BERTScore-F1**, consistently outperforming zero-shot baselines by effectively anchoring responses in retrieved legal statutes and precedents.

### Key Components

- `agentic_rag_core.py`: The heart of the system. Implements the LangGraph state machine, nodes (decomposer, planner, retriever, evaluator, generator), and conditional routing logic.
- `Agentic_rag_app.py`: A Streamlit-based frontend for interacting with the agent, complete with reasoning trace visualization.
- `Qwen3_4B_Finetune.ipynb`: Notebook documenting the fine-tuning workflow (PEFT/QLoRA) applied to enhance domain vocabulary and reasoning.
- `evaluate_rag.py`: Scripts used for performance benchmarking and BERTScore calculation.

### Performance

The system addresses the common pitfalls of vanilla RAG in the legal domain (e.g., retrieval failure on multi-faceted legal questions) by forcing the model to verify evidence sufficiency before final synthesis. This iterative feedback loop significantly reduces hallucinations and increases citation accuracy.

---
*Developed as part of an effort to modernize access to Indian legal information through high-performance natural language understanding and agentic workflows.*
