"""
main.py — The Naive RAG Live Demonstration & Stage Walkthrough.

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
    python main.py --query "What is the coolant temperature for the fusion reactor?"

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
