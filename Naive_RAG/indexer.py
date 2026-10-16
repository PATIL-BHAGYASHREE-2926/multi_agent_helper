"""
indexer.py — The Naive RAG Offline Indexing Pipeline.

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
import os
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
    chunk_size: int = 350,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Stage 2: Chunking (Fixed-size sliding window)
    Splits the continuous text string into overlapping discrete passages.

    [TEACHING NOTE]
    Why sliding window overlap?
    If a crucial sentence (like a secret code or a temperature threshold) spans across
    the boundary between chunk 1 and chunk 2, a hard split might cut the sentence in half!
    An overlap (e.g., 50 characters) guarantees that border context is preserved in both chunks.
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
        # Generate reproducible pseudo-vector based on character values
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
