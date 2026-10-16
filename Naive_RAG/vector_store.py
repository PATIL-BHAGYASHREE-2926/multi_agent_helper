"""
vector_store.py — Vector Database Management (Pinecone with Local Fallback).

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
import os
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
