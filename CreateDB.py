import os
import json
import shutil
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress noisy warnings from the huggingface tokenizer
warnings.filterwarnings("ignore", message=".*position_ids.*")

# ─────────────────────────── CONFIGURATION ───────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ACTS_JSON_PATH       = "data/IndianAct.json"
CASES_JSON_PATH      = "data/IndicLegalQA Dataset.json"
ACTS_DB_DIR          = "./chroma_db_acts"
CASES_DB_DIR         = "./chroma_db_cases"

# all-MiniLM-L6-v2 has a hard 256-token context window.
# At ~4 chars/token, 900 chars ≈ 225 tokens — safe headroom.
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 100

# Number of documents sent to Chroma in a single add_documents() call.
# Keeps memory pressure flat for large datasets.
BATCH_SIZE = 500

# Minimum content length — skip near-empty sections that produce
# noisy embeddings and waste vector space.
MIN_CONTENT_LENGTH = 30


# ─────────────────────────── HELPERS ─────────────────────────────────

def clear_existing_databases():
    """Delete stale Chroma directories before a fresh ingest run."""
    print("Step 1: Clearing old databases...")
    for db_path in [ACTS_DB_DIR, CASES_DB_DIR]:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print(f"  - Deleted: {db_path}")
        else:
            print(f"  - Already clean: {db_path}")


def load_json(filepath: str) -> list:
    """Load and return a JSON array from disk with a clear error on missing file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Missing file: {filepath}\n"
            "Please ensure it is placed inside the data/ folder."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  - Loaded {len(data):,} records from {filepath}")
    return data


def batch_add_to_chroma(db: Chroma, docs: list[Document]) -> None:
    """
    Add documents to an existing Chroma collection in fixed-size batches.
    Prevents OOM errors on large datasets where from_documents() would
    try to embed everything in a single pass.
    """
    total = len(docs)
    for start in range(0, total, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, total)
        batch = docs[start:end]
        db.add_documents(batch)
        print(f"    Indexed {end:,} / {total:,} documents...", end="\r")
    print()  # newline after progress line


# ─────────────────────────── PROCESSING ──────────────────────────────

def process_acts(embedding_model: HuggingFaceEmbeddings) -> None:
    """
    Ingest IndianAct.json into the acts Chroma collection.

    Each raw record (act_title + section + law text) is:
      1. Cleaned of the redundant act-title header on line 1.
      2. Split into overlapping chunks that fit within the 256-token
         context window of all-MiniLM-L6-v2.
      3. Tagged with source metadata for filtered retrieval.
    """
    print("\nStep 2: Processing Statutory Acts...")
    acts_data = load_json(ACTS_JSON_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs: list[Document] = []
    skipped = 0

    for item in acts_data:
        act_title = item.get("act_title", "Unknown Act").strip()
        section   = str(item.get("section", "")).strip()
        law_text  = item.get("law", "").strip()

        # ── Strip the redundant act-name header from line 1 ──────────
        lines = law_text.splitlines()
        if lines and act_title.lower() in lines[0].lower():
            law_text = "\n".join(lines[1:]).strip()

        # ── Skip near-empty or corrupt entries ────────────────────────
        if len(law_text) < MIN_CONTENT_LENGTH:
            skipped += 1
            continue

        # ── Prefix provides context even for mid-section chunks ───────
        prefixed = f"[{act_title} | Section {section}]\n{law_text}"

        # ── Chunk — long sections produce multiple Documents ──────────
        chunks = splitter.split_text(prefixed)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "act_name":    act_title,
                        "section":     section,
                        "chunk_index": i,
                        "source_type": "statute",
                    },
                )
            )

    print(f"  - {len(acts_data):,} raw records → {len(docs):,} chunks "
          f"({skipped} skipped as too short)")
    print(f"  - Writing to {ACTS_DB_DIR} in batches of {BATCH_SIZE}...")

    # Create the collection with the first batch, then add the rest
    first_batch = docs[:BATCH_SIZE]
    db = Chroma.from_documents(
        documents=first_batch,
        embedding=embedding_model,
        persist_directory=ACTS_DB_DIR,
    )
    if len(docs) > BATCH_SIZE:
        batch_add_to_chroma(db, docs[BATCH_SIZE:])

    print("  ✓ Acts database built.")


def process_cases(embedding_model: HuggingFaceEmbeddings) -> None:
    """
    Ingest the IndicLegalQA dataset into the cases Chroma collection.

    Each record is one Q&A pair about a specific case.  These are already
    naturally short (one question + one answer), so we only chunk the rare
    entries whose combined text exceeds the token limit.
    """
    print("\nStep 3: Processing Case Law Q&A...")
    cases_data = load_json(CASES_JSON_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs: list[Document] = []
    skipped = 0

    for item in cases_data:
        case_name      = item.get("case_name", "Unknown").strip()
        judgement_date = item.get("judgement_date", "").strip()
        question       = item.get("question", "").strip()
        answer         = item.get("answer", "").strip()

        content = (
            f"Case Name: {case_name}\n"
            f"Judgement Date: {judgement_date}\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

        if len(content) < MIN_CONTENT_LENGTH:
            skipped += 1
            continue

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "case_name":       case_name,
                        "judgement_date":  judgement_date,
                        "chunk_index":     i,
                        "source_type":     "case_law",
                    },
                )
            )

    print(f"  - {len(cases_data):,} raw records → {len(docs):,} chunks "
          f"({skipped} skipped as too short)")
    print(f"  - Writing to {CASES_DB_DIR} in batches of {BATCH_SIZE}...")

    first_batch = docs[:BATCH_SIZE]
    db = Chroma.from_documents(
        documents=first_batch,
        embedding=embedding_model,
        persist_directory=CASES_DB_DIR,
    )
    if len(docs) > BATCH_SIZE:
        batch_add_to_chroma(db, docs[BATCH_SIZE:])

    print("  ✓ Cases database built.")


# ─────────────────────────── ENTRY POINT ─────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("   VECTOR DATABASE INGESTION")
    print("=" * 50)

    clear_existing_databases()

    # Load once — avoid paying the model-load cost twice
    print("\nLoading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("  ✓ Model ready.")

    try:
        process_acts(embedding_model)
    except FileNotFoundError as e:
        print(f"  [SKIPPED] {e}")

    try:
        process_cases(embedding_model)
    except FileNotFoundError as e:
        print(f"  [SKIPPED] {e}")

    print("\n" + "=" * 50)
    print("   INGESTION COMPLETE")
    print("=" * 50)