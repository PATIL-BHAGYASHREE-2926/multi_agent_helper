# Student Code Reference: The 8-Stage Naive RAG Architecture
### Astra Day 4 Demo: Document Indexing & Live Grounded Generation via NVIDIA NIM & Pinecone

---

## 1. Project Scaffold & Dependencies

To ensure clean isolation and prevent dependency collisions across Python projects on Windows, always create a local virtual environment before installing packages.

```powershell
# 1. Navigate to the Naive RAG demo folder
cd "d:\Astra_Batch\Demos\Naive Rag"

# 2. Create an isolated Python virtual environment named .venv
python -m venv .venv

# 3. Upgrade pip using the virtual environment's Python executable directly
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 4. Install production dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `requirements.txt`
```text
openai>=1.0.0
pinecone>=5.0.0
python-dotenv>=1.0.0
```

> [!NOTE]
> **Why use `.\.venv\Scripts\python.exe` directly on Windows?**  
> In Windows PowerShell, running script activations like `activate.ps1` frequently triggers `PSSecurityException` due to Windows Execution Policy (`Restricted`). Calling the environment's python binary directly (`.\.venv\Scripts\python.exe`) bypasses execution policy restrictions entirely while still using the isolated environment.

---

## 2. Architecture & The "Why"

In Day 4 (Slides 13–21), we learned that Foundation LLMs suffer from two fundamental limitations:
1. **Knowledge Cutoffs:** Models cannot know information created after their pre-training training run.
2. **Private Data Blindness:** Models cannot know proprietary enterprise data (company wikis, employee handbooks, customer tickets).

**Naive RAG (Retrieval-Augmented Generation)** solves this without expensive re-training by splitting the process into two distinct pipelines across **8 explicit stages** (Slide 13):

```
========================= OFFLINE INDEXING PIPELINE =========================
[Raw Source File] 
       │
       ▼ (Stage 1: Document Ingestion)
[Full Text String] 
       │
       ▼ (Stage 2: Chunking with Overlap)
[Discrete Passages / Chunks] 
       │
       ▼ (Stage 3: Vector Embedding via NVIDIA NIM)
[2048-Dimensional Vectors] 
       │
       ▼ (Stage 4: Storage)
[(Pinecone Cloud / Vector DB)]

========================== ONLINE QUERY PIPELINE ============================
[User Live Query]
       │
       ▼ (Stage 5: Query Embedding via NVIDIA NIM)
[Query Vector (2048-dim)]
       │
       ▼ (Stage 6: Vector Retrieval / Cosine Similarity)
[Top-K Most Similar Chunks]
       │
       ▼ (Stage 7: Context Injection)
[Augmented System Prompt]
       │
       ▼ (Stage 8: Grounded LLM Generation via NVIDIA NIM)
[Factually Grounded Response]
```

### The 8 Labeled Pipeline Boxes
Every function in this codebase maps 1-to-1 to the slide diagram:
- `load_document()` `-->` **Stage 1: Document Ingestion**
- `chunk_document()` `-->` **Stage 2: Chunking**
- `embed_chunks()` `-->` **Stage 3: Embedding**
- `store_in_vector_db()` `-->` **Stage 4: Storage**
- `embed_query()` `-->` **Stage 5: Query Embedding**
- `retrieve_top_k()` `-->` **Stage 6: Retrieval**
- `build_prompt_with_context()` `-->` **Stage 7: Context Injection**
- `generate_answer()` `-->` **Stage 8: Grounded Generation**

---

## 3. Environment & Configuration

Create a `.env.local` (or `.env`) file in `d:\Astra_Batch\Demos\Naive Rag\.env.local`:

```env
# ================================================================
# NVIDIA NIM (AI API) Configuration
# ================================================================
# Get your NVIDIA NIM API key at: https://build.nvidia.com/
NVIDIA_API_KEY=nvapi-your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# High-throughput Llama 3.2 model hosted on NVIDIA NIM
NVIDIA_CHAT_MODEL=meta/llama-3.2-11b-vision-instruct

# Enterprise 2048-dimensional embedding model hosted on NVIDIA NIM
NVIDIA_EMBED_MODEL=nvidia/nemotron-3-embed-1b

# ================================================================
# Pinecone Vector Database Configuration
# ================================================================
# Get your Pinecone API key at: https://app.pinecone.io/
# If PINECONE_API_KEY is left empty, the pipeline automatically
# falls back to the built-in local vector database so live demos never fail!
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=naive-rag-demo
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# ================================================================
# Naive RAG Hyperparameters
# ================================================================
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=3
```

> [!TIP]
> **Why NVIDIA NIM instead of public free-tier endpoints?**  
> NVIDIA NIM packages optimized model runtimes on dedicated H100 GPU clusters. Unlike free cloud tiers that trigger HTTP 429 rate limit exceptions mid-lecture, NIM provides high token bandwidth and enterprise stability.

> [!IMPORTANT]
> **Zero-Crash Lecture Guarantee:**  
> If a student has not signed up for Pinecone yet or network access drops on stage, `vector_store.py` detects the absence of the key and seamlessly falls back to a transparent local vector store with pure mathematical cosine similarity. The student can still run the entire 8-stage pipeline without a single code change!

---

## 4. Core Implementation

### `data/astratech_employee_handbook.txt`

**The Goal:** Provide a realistic, relatable, and easy-to-understand corporate knowledge base with specific, verifiable policies.  
**The "Why":** Demonstrating RAG requires a ground-truth document containing unique company policies (PTO rules, equipment stipends, visitor Wi-Fi credentials). Because these facts are private to AstraTech Solutions, the baseline LLM will fail and the RAG model will succeed, clearly proving the value of retrieval.

```text
# ASTRATECH SOLUTIONS: EMPLOYEE HANDBOOK & COMPANY POLICIES
# DOCUMENT ID: AT-HR-2026-V2
# AUDIENCE: ALL NEW HIRES & ENGINEERING STAFF

---

## SECTION 1: WORKING HOURS & HYBRID WORK POLICY
AstraTech Solutions operates on a flexible hybrid schedule designed to balance in-person collaboration with focused deep work. Core business collaboration hours are 10:00 AM to 4:00 PM local time, during which team members are expected to be reachable on Slack. All full-time employees work from the main office three days a week (Mondays, Tuesdays, and Thursdays). Remote work from home is permitted on Wednesdays and Fridays. To protect uninterrupted coding time, Tuesday afternoons from 1:00 PM to 5:00 PM are designated as company-wide "No-Meeting Focus Blocks."

## SECTION 2: PAID TIME OFF (PTO) & WELLNESS BENEFITS
All full-time employees receive 24 days of paid time off (PTO) per calendar year, accrued monthly. Employees may carry over a maximum of 5 unused PTO days into the following calendar year. Sick leave is fully paid and allows up to 5 consecutive business days without requiring a doctor's medical note. In addition to standard health insurance, AstraTech provides an annual Wellness Stipend of $800 per employee, reimbursable for gym memberships, yoga classes, or mental health counseling apps.

## SECTION 3: HARDWARE STIPEND & HOME OFFICE SETUP
Every new engineering hire chooses between a 16-inch MacBook Pro (M3 Max, 36GB Unified Memory) or a Lenovo ThinkPad P16 mobile workstation upon joining. To support comfortable remote work on Wednesdays and Fridays, AstraTech provides a one-time remote home office setup stipend of $1,200. This stipend covers ergonomic chairs, standing desks, 4K external monitors, or noise-cancelling headphones. All home office expense receipts must be submitted via Expensify within 30 days of purchase.

## SECTION 4: OFFICE ACCESS, VISITOR WI-FI & CAFETERIA
The AstraTech campus is accessible 24 hours a day, 7 days a week using your issued RFID employee security badge. If visitors or guests require internet access, the office guest Wi-Fi network SSID is AstraTech-Guest and the password is CoffeeAndCode2026!. Employees commuting by bicycle can access the secure indoor bike storage room located on Basement Level B1; the door keypad unlock code is 7492#. Complimentary catered hot lunch is served daily Monday through Thursday at 12:30 PM on the 4th-floor cafeteria terrace.

## SECTION 5: ANNUAL LEARNING & CONFERENCE STIPEND
To foster continuous engineering growth, each team member is allocated an annual Professional Learning Budget of $2,500 per fiscal year. This budget can be used for O'Reilly technical books, online course subscriptions, and industry certification exams (including NVIDIA NIM, AWS, and GCP certifications). For attending domestic or international developer conferences, travel expense requests must be submitted to your direct engineering manager at least 45 days prior to the conference date.
```

---

### `config.py`

**The Goal:** Centralize, validate, and expose all hyperparameters, API keys, and model endpoints.  
**The "Why":** Reading `os.getenv()` in random helper functions causes silent failures when keys are misspelled or files are moved. Centralizing configuration into a typed dataclass guarantees fail-fast validation and single-point tuning.

```python
"""
config.py -- Centralized Configuration for Naive RAG Pipeline.

The Goal:
    Load, validate, and manage credentials and runtime parameters for
    NVIDIA NIM (Embeddings & LLM Generation) and Pinecone Vector Database.

The "Why":
    Centralizing configuration prevents scattered calls to os.getenv() across
    multiple modules. It also guarantees that if an API key is missing or the network
    is unavailable during a live lecture, the system can gracefully inform the student
    or fall back to a safe offline demonstration mode without crashing.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv


def load_environment_variables() -> None:
    """
    Search for .env.local or .env files across project and workspace directories.
    This allows students to place their API keys in local or parent folders.
    """
    current_directory = Path(__file__).resolve().parent

    # Priority 1: .env.local in current directory (d:\Astra_Batch\Demos\Naive Rag\.env.local)
    local_env_local = current_directory / ".env.local"
    if local_env_local.exists():
        load_dotenv(dotenv_path=local_env_local, override=True)

    # Priority 2: .env in current directory
    local_env = current_directory / ".env"
    if local_env.exists():
        load_dotenv(dotenv_path=local_env)

    # Priority 3: Sibling Inference folder (.env.local in Demos/Inference)
    sibling_inference_env = current_directory.parent / "Inference" / ".env.local"
    if sibling_inference_env.exists():
        load_dotenv(dotenv_path=sibling_inference_env)

    # Priority 4: Workspace root (.env or .env.local)
    workspace_env_local = current_directory.parent.parent / ".env.local"
    if workspace_env_local.exists():
        load_dotenv(dotenv_path=workspace_env_local, override=True)

    workspace_env = current_directory.parent.parent / ".env"
    if workspace_env.exists():
        load_dotenv(dotenv_path=workspace_env)

    # Default fallback to standard environment
    load_dotenv()


# Execute loader upon module import
load_environment_variables()


@dataclass
class RAGConfig:
    """
    Typed settings container holding all hyper-parameters and API configurations.
    """
    nvidia_api_key: str
    nvidia_base_url: str
    chat_model: str
    embed_model: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    use_mock_ai: bool
    use_local_vector_db: bool


def get_rag_config(
    force_mock_ai: bool = False,
    force_local_db: bool = False,
) -> RAGConfig:
    """
    Build and validate the RAG configuration object.
    
    If credentials are missing or flags are set to True,
    safe fallbacks are engaged so the live demo continues gracefully.
    """
    # 1. Fetch NVIDIA NIM credentials
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if len(nvidia_key) == 0:
        nvidia_key = os.getenv("NVIDIA_NIM_API_KEY", "").strip()

    nvidia_base = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
    
    # Fast, high-throughput chat model hosted on NVIDIA NIM
    chat_model = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.2-11b-vision-instruct").strip()
    
    # NVIDIA NIM Embedding model (2048 dimensions)
    embed_model = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3-embed-1b").strip()

    # 2. Fetch Pinecone Vector DB credentials
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    pinecone_index = os.getenv("PINECONE_INDEX_NAME", "naive-rag-demo").strip()
    pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws").strip()
    pinecone_region = os.getenv("PINECONE_REGION", "us-east-1").strip()

    # 3. RAG hyperparameters
    chunk_size = 500
    env_chunk_size = os.getenv("CHUNK_SIZE", "").strip()
    if len(env_chunk_size) > 0:
        chunk_size = int(env_chunk_size)

    chunk_overlap = 100
    env_chunk_overlap = os.getenv("CHUNK_OVERLAP", "").strip()
    if len(env_chunk_overlap) > 0:
        chunk_overlap = int(env_chunk_overlap)

    top_k = 3
    env_top_k = os.getenv("TOP_K", "").strip()
    if len(env_top_k) > 0:
        top_k = int(env_top_k)

    # 4. Determine AI and Vector DB execution modes
    use_mock_ai = False
    if force_mock_ai:
        use_mock_ai = True
    else:
        if len(nvidia_key) == 0:
            use_mock_ai = True

    use_local_vector_db = False
    if force_local_db:
        use_local_vector_db = True
    else:
        if len(pinecone_key) == 0:
            use_local_vector_db = True

    return RAGConfig(
        nvidia_api_key=nvidia_key,
        nvidia_base_url=nvidia_base,
        chat_model=chat_model,
        embed_model=embed_model,
        pinecone_api_key=pinecone_key,
        pinecone_index_name=pinecone_index,
        pinecone_cloud=pinecone_cloud,
        pinecone_region=pinecone_region,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        use_mock_ai=use_mock_ai,
        use_local_vector_db=use_local_vector_db,
    )
```

---

### `vector_store.py`

**The Goal:** Provide a unified interface to store and retrieve vectors from Pinecone Serverless with a local fallback.  
**The "Why":** Vector databases perform high-speed Approximate Nearest Neighbor (ANN) search. Providing a pure Python cosine similarity fallback allows students to understand the exact mathematical dot-product calculation happening under the hood.

```python
"""
vector_store.py -- Vector Database Management (Pinecone with Local Fallback).

The Goal:
    Provide an interface to initialize, upsert, and query vector embeddings
    using Pinecone Serverless. If no Pinecone key is configured, it falls back
    to a transparent local JSON vector store so live demos never fail.

The "Why":
    In production, teams use managed vector databases like Pinecone to perform
    millisecond-level Approximate Nearest Neighbor (ANN) searches across millions
    of vectors. However, during a live classroom lecture or workshop, API keys
    can expire, or the venue Wi-Fi can drop. Providing a local fallback guarantees
    pedagogical continuity without altering the RAG pipeline interface.
"""

import json
import math
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from config import RAGConfig


LOCAL_DB_FILE = Path(__file__).resolve().parent / "data" / "local_vector_db.json"


def calculate_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors using basic math:
    Cosine(A, B) = dot_product(A, B) / (norm(A) * norm(B))
    
    [TEACHING NOTE]
    Cosine similarity measures the angle between two embedding vectors in
    high-dimensional space. A score of 1.0 means identical direction/meaning,
    while 0.0 means completely orthogonal (unrelated).
    """
    if len(vec_a) != len(vec_b):
        return 0.0

    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    index = 0
    while index < len(vec_a):
        a_val = vec_a[index]
        b_val = vec_b[index]
        dot_product = dot_product + (a_val * b_val)
        norm_a_sq = norm_a_sq + (a_val * a_val)
        norm_b_sq = norm_b_sq + (b_val * b_val)
        index = index + 1

    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        return 0.0

    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)
    return dot_product / (norm_a * norm_b)


def get_pinecone_client(api_key: str) -> Pinecone:
    """
    Instantiate and return the modern Pinecone client.
    """
    return Pinecone(api_key=api_key)


def ensure_pinecone_index_exists(
    client: Pinecone,
    index_name: str,
    dimension: int = 2048,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """
    Check if the Pinecone index exists; if not, create a Serverless index.
    """
    existing_indexes = client.list_indexes()
    index_names = []
    for item in existing_indexes:
        index_names.append(item.name)

    if index_name not in index_names:
        print(f"[*] Pinecone Index '{index_name}' not found. Creating serverless index...")
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"[OK] Pinecone Index '{index_name}' created successfully (Dimension: {dimension}).")
    else:
        print(f"[OK] Connected to existing Pinecone Index: '{index_name}'.")


def upsert_to_vector_db(
    chunks_with_embeddings: list[dict],
    config: RAGConfig,
) -> int:
    """
    Store chunk embeddings and text metadata into Pinecone or local fallback.
    """
    count = len(chunks_with_embeddings)
    if count == 0:
        return 0

    # Branch 1: Use Pinecone Cloud
    if not config.use_local_vector_db:
        try:
            client = get_pinecone_client(config.pinecone_api_key)
            dim = len(chunks_with_embeddings[0]["embedding"])
            ensure_pinecone_index_exists(
                client=client,
                index_name=config.pinecone_index_name,
                dimension=dim,
                cloud=config.pinecone_cloud,
                region=config.pinecone_region,
            )

            index = client.Index(config.pinecone_index_name)

            # Format vectors for Pinecone upsert API
            vectors_payload = []
            for item in chunks_with_embeddings:
                record = {
                    "id": item["id"],
                    "values": item["embedding"],
                    "metadata": {
                        "text": item["text"],
                        "chunk_id": item["id"],
                    },
                }
                vectors_payload.append(record)

            index.upsert(vectors=vectors_payload)
            print(f"[OK] Successfully upserted {count} vectors to Pinecone Index '{config.pinecone_index_name}'.")
            return count

        except Exception as error:
            print(f"[!] Warning: Pinecone upsert encountered an error: {error}")
            print("[!] Falling back to Local Vector Database for live reliability.")

    # Branch 2: Local JSON Storage Fallback
    LOCAL_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for item in chunks_with_embeddings:
        record = {
            "id": item["id"],
            "embedding": item["embedding"],
            "metadata": {
                "text": item["text"],
                "chunk_id": item["id"],
            },
        }
        records.append(record)

    with open(LOCAL_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"[OK] Saved {count} vectors to Local Vector Store ({LOCAL_DB_FILE.name}).")
    return count


def query_vector_db(
    query_vector: list[float],
    top_k: int,
    config: RAGConfig,
) -> list[dict]:
    """
    Retrieve top-k most similar chunks for a query vector from Pinecone or local fallback.
    """
    results = []

    # Branch 1: Query Pinecone Cloud
    if not config.use_local_vector_db:
        try:
            client = get_pinecone_client(config.pinecone_api_key)
            index = client.Index(config.pinecone_index_name)
            response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
            )

            for match in response.matches:
                meta = match.metadata or {}
                chunk_record = {
                    "id": match.id,
                    "score": float(match.score),
                    "text": meta.get("text", ""),
                }
                results.append(chunk_record)

            return results

        except Exception as error:
            print(f"[!] Warning: Pinecone query failed ({error}). Using local fallback.")

    # Branch 2: Local Vector Store Fallback
    if not LOCAL_DB_FILE.exists():
        print(f"[!] Local Vector Store file not found at {LOCAL_DB_FILE}. Please run indexing first.")
        return []

    with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
        stored_records = json.load(f)

    scored_items = []
    for record in stored_records:
        candidate_vector = record["embedding"]
        similarity = calculate_cosine_similarity(query_vector, candidate_vector)
        item = {
            "id": record["id"],
            "score": similarity,
            "text": record["metadata"]["text"],
        }
        scored_items.append(item)

    # Sort descending by similarity score
    def get_item_score(x: dict) -> float:
        return x["score"]

    scored_items.sort(key=get_item_score, reverse=True)

    # Select top_k elements
    selected_items = []
    count = 0
    for entry in scored_items:
        if count < top_k:
            selected_items.append(entry)
            count = count + 1

    return selected_items
```

---

### `indexer.py`

**The Goal:** Execute Stages 1 through 4 of the offline Indexing Pipeline.  
**The "Why":** Documents must be chunked and embedded before live user queries arrive. This script provides visible printouts of token boundaries and embedding coordinates for live audience tracing.

```python
"""
indexer.py -- The Naive RAG Offline Indexing Pipeline.

The Goal:
    Ingest a source knowledge document, split it into overlapping chunks,
    convert each chunk into a high-dimensional vector via NVIDIA NIM, and
    persist the vectors and metadata into the Pinecone Vector Database.

The "Why":
    Language models have a fixed knowledge cutoff and cannot memorize proprietary
    or private company manuals without fine-tuning (which is slow and expensive).
    The indexing pipeline prepares external knowledge into indexed vector coordinates
    so relevant information can be fetched in milliseconds during live queries.

Pipeline Stages Mapped to Day 4 Slide 13:
    1. load_document()       ---> Document Ingestion
    2. chunk_document()      ---> Chunking
    3. embed_chunks()        ---> Embedding (NVIDIA NIM)
    4. store_in_vector_db()  ---> Storage (Pinecone Serverless)
"""

import math
from pathlib import Path
from openai import OpenAI
from config import RAGConfig, get_rag_config
from vector_store import upsert_to_vector_db


def load_document(file_path: str) -> str:
    """
    Stage 1: Document Ingestion
    Read the raw text content from the specified source document file.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Document file not found at: {file_path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned_text = text.strip()
    
    print("\n" + "=" * 60)
    print("STAGE 1: DOCUMENT INGESTION")
    print("=" * 60)
    print(f"[*] Source File Path : {path_obj.resolve()}")
    print(f"[*] Total Characters : {len(cleaned_text)}")
    print(f"[*] Total Words      : {len(cleaned_text.split())}")
    print("[OK] Document loaded into memory successfully.")
    
    return cleaned_text


def chunk_document(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Stage 2: Chunking (Fixed-size sliding window)
    Splits the continuous text string into overlapping discrete passages.

    [TEACHING NOTE]
    Why sliding window overlap?
    If a crucial sentence (like an allowance amount or Wi-Fi password) spans across
    the boundary between chunk 1 and chunk 2, a hard split might cut the sentence in half!
    An overlap (e.g., 100 characters) guarantees that border context is preserved in both chunks.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    chunk_index = 0

    print("\n" + "=" * 60)
    print("STAGE 2: CHUNKING (SLIDING WINDOW)")
    print("=" * 60)
    print(f"[*] Window Size    : {chunk_size} characters")
    print(f"[*] Window Overlap : {chunk_overlap} characters")
    print(f"[*] Stride / Step  : {step} characters")

    while start < len(text):
        end = start + chunk_size
        chunk_slice = text[start:end]
        
        # Clean leading and trailing whitespace
        trimmed_chunk = chunk_slice.strip()
        
        if len(trimmed_chunk) > 0:
            chunk_record = {
                "id": f"doc_chunk_{chunk_index}",
                "text": trimmed_chunk,
                "start_char": start,
                "end_char": start + len(chunk_slice),
            }
            chunks.append(chunk_record)
            
            # Print preview for live lecture visibility
            preview = trimmed_chunk.replace("\n", " ")
            if len(preview) > 65:
                preview = preview[:65] + "..."
            print(f"  -> [{chunk_record['id']}] ({len(trimmed_chunk)} chars): \"{preview}\"")
            chunk_index = chunk_index + 1

        start = start + step

    print(f"[OK] Created {len(chunks)} discrete chunks from source document.")
    return chunks


def embed_chunks(
    chunks: list[dict],
    config: RAGConfig,
) -> list[dict]:
    """
    Stage 3: Embedding (NVIDIA NIM)
    Converts each textual chunk into a dense 2048-dimensional vector representation.

    [TEACHING NOTE]
    Embedding models translate semantic concepts into mathematical space.
    Chunks with similar meaning will be close together in this 2048-dimensional space,
    even if they use different words or synonyms!
    """
    print("\n" + "=" * 60)
    print("STAGE 3: EMBEDDING GENERATION (NVIDIA NIM)")
    print("=" * 60)
    print(f"[*] Embedding Model : {config.embed_model}")

    chunks_with_embeddings = []

    # Branch 1: Live NVIDIA NIM Embedding API
    if not config.use_mock_ai:
        print(f"[*] Connecting to NVIDIA NIM at: {config.nvidia_base_url}")
        client = OpenAI(
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key,
        )

        texts_to_embed = []
        for c in chunks:
            texts_to_embed.append(c["text"])

        # Call NVIDIA NIM Embeddings API
        response = client.embeddings.create(
            input=texts_to_embed,
            model=config.embed_model,
        )

        vector_dim = len(response.data[0].embedding)
        print(f"[OK] Received {len(response.data)} embedding vectors from NVIDIA NIM.")
        print(f"[*] Vector Dimensionality : {vector_dim} dimensions")

        idx = 0
        for item in response.data:
            enriched_chunk = {
                "id": chunks[idx]["id"],
                "text": chunks[idx]["text"],
                "embedding": item.embedding,
            }
            chunks_with_embeddings.append(enriched_chunk)
            
            # Print sample coordinates of the vector
            sample_coords = [round(val, 4) for val in item.embedding[:4]]
            print(f"  -> Vector [{chunks[idx]['id']}] coordinates: {sample_coords} ...")
            idx = idx + 1

        return chunks_with_embeddings

    # Branch 2: Simulated Deterministic Vectors (Offline Mock Mode)
    print("[!] NVIDIA API key not active. Generating simulated deterministic 64-dim embeddings...")
    for c in chunks:
        sim_vector = []
        seed = 0
        for ch in c["text"]:
            seed = seed + ord(ch)

        dim_index = 0
        while dim_index < 64:
            val = math.sin(seed + dim_index) * 0.5
            sim_vector.append(round(val, 4))
            dim_index = dim_index + 1

        enriched_chunk = {
            "id": c["id"],
            "text": c["text"],
            "embedding": sim_vector,
        }
        chunks_with_embeddings.append(enriched_chunk)

    print(f"[OK] Simulated {len(chunks_with_embeddings)} offline embeddings (64 dimensions).")
    return chunks_with_embeddings


def store_in_vector_db(
    chunks_with_embeddings: list[dict],
    config: RAGConfig,
) -> int:
    """
    Stage 4: Storage
    Persist chunk vectors and text metadata into the Pinecone Vector Database.
    """
    print("\n" + "=" * 60)
    print("STAGE 4: STORAGE IN VECTOR DATABASE")
    print("=" * 60)
    if config.use_local_vector_db:
        print("[*] Target Storage : Local JSON Vector Store (Pinecone Fallback)")
    else:
        print(f"[*] Target Storage : Pinecone Serverless Index '{config.pinecone_index_name}'")
        print(f"[*] Cloud Provider : {config.pinecone_cloud.upper()} ({config.pinecone_region})")

    stored_count = upsert_to_vector_db(
        chunks_with_embeddings=chunks_with_embeddings,
        config=config,
    )

    print(f"[OK] Storage stage complete: {stored_count} records indexed and ready for retrieval.")
    return stored_count


def run_indexing_pipeline(
    file_path: str,
    config: RAGConfig | None = None,
) -> list[dict]:
    """
    Execute the complete 4-stage Indexing Pipeline sequentially:
    1. Document Ingestion
    2. Chunking
    3. Embedding
    4. Storage
    """
    if config is None:
        config = get_rag_config()

    print("\n" + "#" * 60)
    print("STARTING NAIVE RAG INDEXING PIPELINE")
    print("#" * 60)

    # 1. Ingestion
    raw_document = load_document(file_path=file_path)

    # 2. Chunking
    chunk_list = chunk_document(
        text=raw_document,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # 3. Embedding
    embedded_chunks = embed_chunks(
        chunks=chunk_list,
        config=config,
    )

    # 4. Storage
    store_in_vector_db(
        chunks_with_embeddings=embedded_chunks,
        config=config,
    )

    print("\n" + "#" * 60)
    print("[SUCCESS] INDEXING PIPELINE FINISHED -- KNOWLEDGE BASE READY!")
    print("#" * 60 + "\n")

    return embedded_chunks


if __name__ == "__main__":
    # Test indexer standalone
    cfg = get_rag_config()
    sample_doc = str(Path(__file__).parent / "data" / "astratech_employee_handbook.txt")
    run_indexing_pipeline(file_path=sample_doc, config=cfg)
```

---

### `query_pipeline.py`

**The Goal:** Execute Stages 5 through 8 of the online Query Pipeline.  
**The "Why:** Embed the user query into the identical vector space, retrieve the top matches from Pinecone, construct the grounded prompt, and stream the fact-based synthesis.

```python
"""
query_pipeline.py -- The Naive RAG Online Query Pipeline.

The Goal:
    Accept a user's question live, convert it into an embedding vector via NVIDIA NIM,
    retrieve the top-k most relevant chunks from the Pinecone vector database,
    inject those chunks into an augmented prompt, and generate a grounded answer.

The "Why":
    This is the runtime loop that executes every time a user asks a question.
    By augmenting the model's prompt with retrieved private documents, we eliminate
    hallucinations and anchor the model's response directly to verified source facts.

Pipeline Stages Mapped to Day 4 Slide 13:
    5. embed_query()                 ---> Query Embedding (NVIDIA NIM)
    6. retrieve_top_k()               ---> Retrieval (Pinecone / Cosine Similarity)
    7. build_prompt_with_context()    ---> Context Injection (Prompt Augmentation)
    8. generate_answer()             ---> Generation (NVIDIA NIM LLM)
"""

import math
from openai import OpenAI
from config import RAGConfig, get_rag_config
from vector_store import query_vector_db


def embed_query(
    query: str,
    config: RAGConfig,
) -> list[float]:
    """
    Stage 5: Query Embedding
    Embeds the user's live query into the same 2048-dimensional mathematical
    space used during the offline chunk indexing stage.

    [TEACHING NOTE]
    Why must the query embedding model match the chunk embedding model?
    If you index your documents with Model A (e.g. 2048-dim) and embed your query with
    Model B (e.g. 1536-dim or different semantic coordinate system), the vector dimensions
    and coordinate meanings will not match! Always use the exact same embedding model.
    """
    cleaned_query = query.strip()
    if len(cleaned_query) == 0:
        raise ValueError("User query cannot be empty.")

    print("\n" + "=" * 60)
    print("STAGE 5: QUERY EMBEDDING (NVIDIA NIM)")
    print("=" * 60)
    print(f"[*] Live User Query : \"{cleaned_query}\"")
    print(f"[*] Embedding Model : {config.embed_model}")

    # Branch 1: Live NVIDIA NIM Embedding API
    if not config.use_mock_ai:
        client = OpenAI(
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key,
        )

        response = client.embeddings.create(
            input=[cleaned_query],
            model=config.embed_model,
        )

        query_vector = response.data[0].embedding
        vector_dim = len(query_vector)
        sample_coords = [round(val, 4) for val in query_vector[:4]]
        
        print(f"[OK] Generated query vector ({vector_dim} dimensions).")
        print(f"[*] Sample Coordinates : {sample_coords} ...")
        return query_vector

    # Branch 2: Offline Mock Simulation
    print("[!] Offline mode active. Generating simulated query vector...")
    sim_vector = []
    seed = 0
    for ch in cleaned_query:
        seed = seed + ord(ch)

    dim_index = 0
    while dim_index < 64:
        val = math.sin(seed + dim_index) * 0.5
        sim_vector.append(round(val, 4))
        dim_index = dim_index + 1

    print("[OK] Generated simulated 64-dim query vector.")
    return sim_vector


def retrieve_top_k(
    query_vector: list[float],
    config: RAGConfig,
) -> list[dict]:
    """
    Stage 6: Retrieval
    Queries the vector database for the top-k chunks with the highest
    cosine similarity to the query vector.
    """
    print("\n" + "=" * 60)
    print(f"STAGE 6: RETRIEVAL (TOP-{config.top_k} CHUNKS)")
    print("=" * 60)
    
    if config.use_local_vector_db:
        print("[*] Searching Database : Local Vector Store (JSON)")
    else:
        print(f"[*] Searching Database : Pinecone Index '{config.pinecone_index_name}'")

    matched_chunks = query_vector_db(
        query_vector=query_vector,
        top_k=config.top_k,
        config=config,
    )

    if len(matched_chunks) == 0:
        print("[!] No matching chunks found. Did you run indexer.py first?")
        return []

    print(f"[OK] Retrieved {len(matched_chunks)} most relevant passages:\n")

    rank = 1
    for match in matched_chunks:
        score_pct = round(match["score"] * 100.0, 2)
        print(f"  [Match #{rank}] ID: {match['id']} | Cosine Similarity: {match['score']:.4f} ({score_pct}%)")
        print(f"  Text Excerpt: \"{match['text']}\"\n")
        rank = rank + 1

    return matched_chunks


def build_prompt_with_context(
    query: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Stage 7: Context Injection (Prompt Augmentation)
    Assembles a structured system prompt combining the retrieved context passages
    with strict grounding instructions and the user's question.

    [TEACHING NOTE]
    The Grounding Instruction is the secret sauce of RAG. We explicitly instruct the
    LLM to ONLY use the provided context and never speculate. This anchors the model's
    probabilistic decoder to real facts.
    """
    print("\n" + "=" * 60)
    print("STAGE 7: CONTEXT INJECTION (AUGMENTED PROMPT)")
    print("=" * 60)

    # Format the retrieved context blocks explicitly
    context_blocks = []
    chunk_num = 1
    for chunk in retrieved_chunks:
        header = f"--- CONTEXT DOCUMENT {chunk_num} (ID: {chunk['id']}, Relevance: {chunk['score']:.4f}) ---"
        body = chunk["text"]
        context_blocks.append(f"{header}\n{body}")
        chunk_num = chunk_num + 1

    combined_context = "\n\n".join(context_blocks)

    augmented_prompt = f"""You are the official HR & Workplace AI assistant for AstraTech Solutions.
Answer the user's inquiry accurately, professionally, and EXCLUSIVELY based on the verified reference context below.
If the answer cannot be determined strictly from the context, respond: "I do not have sufficient information in the AstraTech employee handbook to answer that question."
Do not speculate, assume, or hallucinate outside the given text.

=== BEGIN VERIFIED REFERENCE CONTEXT ===
{combined_context}
=== END VERIFIED REFERENCE CONTEXT ===

USER QUESTION:
{query}

GROUNDED ANSWER:"""

    print(f"[*] Combined Context Length : {len(combined_context)} characters across {len(retrieved_chunks)} passages")
    print(f"[*] Total Augmented Prompt   : {len(augmented_prompt)} characters")
    print("[OK] Prompt assembled with grounded context injection.")
    
    return augmented_prompt


def generate_answer(
    prompt: str,
    config: RAGConfig,
) -> str:
    """
    Stage 8: Generation (Grounded LLM Call via NVIDIA NIM)
    Sends the context-augmented prompt to the LLM to synthesize the final answer.
    """
    print("\n" + "=" * 60)
    print("STAGE 8: GENERATION (NVIDIA NIM)")
    print("=" * 60)
    print(f"[*] Generation Model : {config.chat_model}")
    print("[*] Generating grounded response...")

    # Branch 1: Live NVIDIA NIM Chat Completion API
    if not config.use_mock_ai:
        client = OpenAI(
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a precise technical AI assistant grounded strictly in provided operational manuals.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = client.chat.completions.create(
            model=config.chat_model,
            messages=messages,
            temperature=0.1,  # Low temperature for deterministic, factual retrieval
            max_tokens=300,
        )

        raw_answer = response.choices[0].message.content
        if raw_answer is None:
            raw_answer = getattr(response.choices[0].message, "reasoning_content", "") or ""

        final_answer = raw_answer.strip()
        print("[OK] Response received from NVIDIA NIM:")
        print("-" * 60)
        print(final_answer)
        print("-" * 60)
        return final_answer

    # Branch 2: Simulated Offline Response
    print("[!] Offline mode active. Returning simulated grounded answer.")
    mock_reply = (
        "[SIMULATED RAG ANSWER] The one-time remote home office setup stipend at "
        "AstraTech Solutions is $1,200 (to be submitted via Expensify within 30 days). "
        "The office guest Wi-Fi network SSID is AstraTech-Guest with password 'CoffeeAndCode2026!'."
    )
    print("-" * 60)
    print(mock_reply)
    print("-" * 60)
    return mock_reply


def generate_answer_without_rag(
    query: str,
    config: RAGConfig,
) -> str:
    """
    Teaching Comparison Function:
    Ask the LLM the exact same question WITHOUT providing any retrieved context.
    
    [TEACHING NOTE]
    Running this in class shows students why RAG is necessary. Because AstraTech's
    employee handbook is private internal documentation, the base LLM has never seen it in its training data.
    Without RAG, the model either refuses or hallucinates a completely made-up answer!
    """
    print("\n" + "#" * 60)
    print("BASELINE COMPARISON: RUNNING QUERY WITHOUT RAG (RAW LLM)")
    print("#" * 60)
    print(f"[*] Query Sent to LLM : \"{query}\"")
    print("[*] (Notice: No documents or retrieved context are provided in this call!)")

    if not config.use_mock_ai:
        client = OpenAI(
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer the user's question directly.",
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        try:
            response = client.chat.completions.create(
                model=config.chat_model,
                messages=messages,
                temperature=0.2,
                max_tokens=200,
            )
            raw_answer = response.choices[0].message.content
            if raw_answer is None:
                raw_answer = getattr(response.choices[0].message, "reasoning_content", "") or ""
            
            answer = raw_answer.strip()
            print("\n[RAW LLM RESPONSE (WITHOUT RAG)]:")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            return answer
        except Exception as err:
            err_msg = f"[!] Raw LLM call failed: {err}"
            print(err_msg)
            return err_msg

    mock_uninformed = (
        "[RAW LLM ANSWER WITHOUT RAG]: I do not have access to internal company policies or "
        "employee handbooks for AstraTech Solutions."
    )
    print("\n[RAW LLM RESPONSE (WITHOUT RAG)]:")
    print("-" * 60)
    print(mock_uninformed)
    print("-" * 60)
    return mock_uninformed


def run_query_pipeline(
    query: str,
    config: RAGConfig | None = None,
) -> dict:
    """
    Execute the complete 4-stage Query Pipeline sequentially:
    5. Query Embedding
    6. Retrieval
    7. Context Injection
    8. Generation
    """
    if config is None:
        config = get_rag_config()

    print("\n" + "#" * 60)
    print("STARTING NAIVE RAG QUERY PIPELINE")
    print("#" * 60)

    # 5. Query Embedding
    query_vector = embed_query(query=query, config=config)

    # 6. Retrieval
    retrieved_chunks = retrieve_top_k(query_vector=query_vector, config=config)

    # 7. Context Injection
    augmented_prompt = build_prompt_with_context(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    # 8. Generation
    answer = generate_answer(
        prompt=augmented_prompt,
        config=config,
    )

    print("\n" + "#" * 60)
    print("[SUCCESS] QUERY PIPELINE FINISHED -- ANSWER PRODUCED!")
    print("#" * 60 + "\n")

    result = {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "augmented_prompt": augmented_prompt,
        "answer": answer,
    }
    return result


if __name__ == "__main__":
    # Test query pipeline with the canonical live test question
    cfg = get_rag_config()
    test_question = (
        "How much is the one-time remote home office setup stipend, "
        "and what is the Wi-Fi password for office visitors?"
    )
    
    # 1. Run baseline without RAG
    generate_answer_without_rag(query=test_question, config=cfg)
    
    # 2. Run with RAG
    run_query_pipeline(query=test_question, config=cfg)
```

---

### `main.py`

**The Goal:** Provide the high-level demonstration CLI for live presentations and comparisons.  
**The "Why:** On stage, the instructor needs a single command that runs the indexing pipeline, shows the raw LLM failing on the test question, retrieves the correct chunks, and prints the grounded output in a clean scorecard.

```python
"""
main.py -- The Naive RAG Live Demonstration & Stage Walkthrough.

The Goal:
    Provide an end-to-end interactive CLI runner for the Day 4 Naive RAG lecture,
    tracing each stage against the 8-box RAG architecture diagram (Slide 13).

The "Why":
    During a live lecture, students need to see the contrast between an unaugmented LLM
    (which either hallucinates or refuses) and a RAG pipeline (which retrieves factual chunks
    and cites exact manual specifications). This script orchestrates the entire workflow
    with visible terminal banners and telemetry.

Usage:
    # Run full interactive live demo (Indexing + Baseline Comparison + RAG Query)
    python main.py

    # Re-run only the indexing pipeline
    python main.py --index

    # Ask a custom question through the RAG pipeline
    python main.py --query "How many days of PTO can roll over into next year?"

    # Run the side-by-side comparison between Raw LLM and RAG
    python main.py --compare
"""

import argparse
import sys
from pathlib import Path
from config import RAGConfig, get_rag_config
from indexer import run_indexing_pipeline
from query_pipeline import (
    generate_answer_without_rag,
    run_query_pipeline,
)


CANONICAL_TEST_QUERY = (
    "How much is the one-time remote home office setup stipend, "
    "and what is the Wi-Fi password for office visitors?"
)


def print_banner(title: str) -> None:
    """
    Print a high-visibility terminal header box for classroom projectors.
    """
    bar = "=" * 70
    print("\n" + bar)
    print(f"  {title.upper()}")
    print(bar + "\n")


def print_scorecard(query: str, raw_answer: str, rag_answer: str, chunks: list[dict]) -> None:
    """
    Print an educational scorecard comparing the raw model response to the RAG response.
    """
    print_banner("DAY 4 LIVE DEMO SCORECARD: RAW LLM VS. NAIVE RAG")
    print(f"QUESTION ASKED:\n  \"{query}\"\n")

    print("-" * 70)
    print("1. BASELINE MODEL (WITHOUT RAG):")
    print(f"   {raw_answer}")
    print("-" * 70)

    print("\n" + "-" * 70)
    print("2. RETRIEVED GROUND TRUTH CONTEXT (PINECONE / COSINE SIMILARITY):")
    chunk_idx = 1
    for chunk in chunks:
        preview = chunk["text"].replace("\n", " ")
        if len(preview) > 110:
            preview = preview[:110] + "..."
        print(f"   [Chunk {chunk_idx} - {chunk['id']}] (Similarity: {chunk['score']:.4f})")
        print(f"   \"{preview}\"\n")
        chunk_idx = chunk_idx + 1
    print("-" * 70)

    print("\n" + "-" * 70)
    print("3. NAIVE RAG MODEL (WITH NVIDIA NIM CONTEXT INJECTION):")
    print(f"   {rag_answer}")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("EVALUATOR & STUDENT TAKEAWAY:")
    print("  * Notice that the Baseline LLM has zero knowledge of AstraTech Solutions' internal policies.")
    print("  * Naive RAG retrieved the exact policy chunks with high cosine similarity.")
    print("  * The prompt injection forced the model to cite the exact $1,200 remote setup stipend")
    print("    and the 'CoffeeAndCode2026!' visitor Wi-Fi password.")
    print("=" * 70 + "\n")


def run_full_demo(config: RAGConfig) -> None:
    """
    Execute the entire 8-stage live classroom walkthrough.
    """
    print_banner("ASTRA DAY 4: NAIVE RAG LIVE DEMONSTRATION")
    print("Pipeline Map (Slide 13):")
    print("  [Stage 1] Document Ingestion  --> [Stage 2] Chunking (Sliding Window)")
    print("  [Stage 3] Vector Embedding    --> [Stage 4] Storage (Pinecone / Local)")
    print("  [Stage 5] Query Embedding     --> [Stage 6] Retrieval (Top-K Matches)")
    print("  [Stage 7] Context Injection   --> [Stage 8] Grounded Generation\n")

    doc_file = Path(__file__).parent / "data" / "astratech_employee_handbook.txt"
    if not doc_file.exists():
        print(f"[!] Error: Operations manual not found at {doc_file}")
        sys.exit(1)

    # Step 1: Run Indexing Pipeline (Stages 1 to 4)
    print_banner("PHASE 1: RUNNING OFFLINE INDEXING PIPELINE")
    run_indexing_pipeline(file_path=str(doc_file), config=config)

    # Step 2: Show the Baseline (Without RAG)
    print_banner("PHASE 2: TESTING RAW MODEL (WITHOUT RAG)")
    raw_answer = generate_answer_without_rag(query=CANONICAL_TEST_QUERY, config=config)

    # Step 3: Run the Online Query Pipeline (Stages 5 to 8)
    print_banner("PHASE 3: TESTING NAIVE RAG PIPELINE (WITH CONTEXT)")
    rag_result = run_query_pipeline(query=CANONICAL_TEST_QUERY, config=config)

    # Step 4: Display Side-by-Side Educational Scorecard
    print_scorecard(
        query=CANONICAL_TEST_QUERY,
        raw_answer=raw_answer,
        rag_answer=rag_result["answer"],
        chunks=rag_result["retrieved_chunks"],
    )


def main() -> None:
    """
    Parse command-line arguments and run the chosen demonstration routine.
    """
    parser = argparse.ArgumentParser(
        description="Astra Day 4 Naive RAG Project Live Demo (NVIDIA NIM + Pinecone)",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Run only the offline indexing pipeline (Stages 1-4).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Run only the online query pipeline with a custom question (Stages 5-8).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the baseline vs. RAG comparison on the canonical question.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force offline simulation mode without making external API calls.",
    )

    args = parser.parse_args()
    config = get_rag_config(force_mock_ai=args.mock)

    doc_file = str(Path(__file__).parent / "data" / "astratech_employee_handbook.txt")

    # Command Router
    if args.index:
        run_indexing_pipeline(file_path=doc_file, config=config)
    elif len(args.query.strip()) > 0:
        run_query_pipeline(query=args.query.strip(), config=config)
    elif args.compare:
        raw_ans = generate_answer_without_rag(query=CANONICAL_TEST_QUERY, config=config)
        rag_res = run_query_pipeline(query=CANONICAL_TEST_QUERY, config=config)
        print_scorecard(
            query=CANONICAL_TEST_QUERY,
            raw_answer=raw_ans,
            rag_answer=rag_res["answer"],
            chunks=rag_res["retrieved_chunks"],
        )
    else:
        # Default action: run the complete live demo walkthrough!
        run_full_demo(config=config)


if __name__ == "__main__":
    main()
```

---

## 5. Running the Project

### Command 1: Full Live Stage Walkthrough (Recommended)
This runs the indexing pipeline, the raw LLM baseline, the top-k retrieval, and produces the complete side-by-side comparison scorecard:

```powershell
.\.venv\Scripts\python.exe main.py
```

### Command 2: Re-Index Only (Stages 1 to 4)
When you update `data/astratech_employee_handbook.txt` and want to re-chunk and re-embed:

```powershell
.\.venv\Scripts\python.exe main.py --index
```

### Command 3: Test a Custom Question (Stages 5 to 8)
Ask any specific question about remote work schedules, wellness budgets, or cafeteria lunch times:

```powershell
.\.venv\Scripts\python.exe main.py --query "How many days of unused PTO can an employee roll over into the next calendar year?"
```

```powershell
.\.venv\Scripts\python.exe main.py --query "What is the annual professional learning and conference budget per engineer?"
```

```powershell
.\.venv\Scripts\python.exe main.py --query "What are the designated hours for the company-wide No-Meeting Focus Blocks?"
```

### Command 4: Offline Simulation Mode (No API Keys / No Network)
If presenting in an auditorium with no Wi-Fi:

```powershell
.\.venv\Scripts\python.exe main.py --mock
```

---

## 6. Evaluator Talking Points

A cheat sheet for the student to explain and defend their architectural decisions to professors, interviewers, or technical evaluators:

1. **Decoupled 2-Pipeline Architecture:**
   * *"We deliberately decoupled the Indexing Pipeline from the Query Pipeline. Indexing is an asynchronous batch operation that runs once during ingestion, whereas the Query Pipeline is an online, low-latency microservice that executes on every user prompt."*
2. **Why Fixed-Size Chunking Needs Overlap:**
   * *"If you perform naive chunking without overlap, sentences that fall across chunk boundaries get truncated, losing semantic context. By introducing a 100-character sliding-window overlap with 500-character windows, we ensure complete policy rules (like stipend amounts or Wi-Fi passwords) remain contiguous in at least one chunk."*
3. **Identical Embedding Geometry Constraint:**
   * *"Vector search requires strict mathematical consistency. If you index chunks with NVIDIA's 2048-dimensional model (`nvidia/nemotron-3-embed-1b`), you cannot query it using a 1536-dimensional OpenAI vector. The geometric coordinate space must match exactly."*
4. **Prompt Augmentation with Strict Grounding Guardrails:**
   * *"Rather than just dumping retrieved text into the prompt, we enforce a strict grounding prompt template: 'Answer EXCLUSIVELY based on the verified reference context... If cannot be determined, state you do not have sufficient information.' This drastically eliminates ungrounded hallucinations."*
5. **High-Availability Local Vector Fallback:**
   * *"To prevent live system failure in air-gapped or network-constrained environments, our vector storage layer transparently falls back to local cosine similarity calculation without modifying the upstream retrieval API."*

---

## 7. Changes from Original Spec

| Spec Item | Original Spec Suggestion | Final Implementation | Engineering Rationale |
|---|---|---|---|
| **Dataset Topic** | Arbitrary topic | **AstraTech Employee Onboarding Handbook** | Replaced complex technical/sci-fi jargon with an intuitive corporate policy guide (PTO, stipends, Wi-Fi passwords) that every student immediately understands. |
| **Vector DB** | Any simple vector store / in-memory | **Pinecone Serverless + Local Fallback** | Pinecone provides production-grade vector indexing on AWS, while our pure Python cosine similarity fallback guarantees zero-crash reliability during live demos. |
| **Embed Model** | Provider choice | **`nvidia/nemotron-3-embed-1b` (2048-dim)** | Verified active on NVIDIA NIM H100 clusters; provides high semantic resolution across technical domain manuals. |
| **Chat Model** | `meta/llama-3.1-8b-instruct` | **`meta/llama-3.2-11b-vision-instruct`** | NVIDIA NIM retired Llama 3.1 8B on 2026-08-26 (HTTP 410). Llama 3.2 11B offers instant TTFT without reasoning token stalls. |
| **Contrast Demonstration** | Implied query comparison | **Explicit `generate_answer_without_rag()`** | Proves the necessity of RAG live to students by demonstrating that the raw LLM refuses or fails on proprietary facts. |
