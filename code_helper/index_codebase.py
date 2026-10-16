import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone, ServerlessSpec
from code_helper.settings import (
    PROJECT_PATH, PINECONE_API_KEY, PINECONE_INDEX_NAME,
    PINECONE_CLOUD, PINECONE_REGION, EMBED_DIM, CHUNK_SIZE, CHUNK_OVERLAP,
)
from code_helper.tools.embeddings import embed_texts


def chunk_text(text, size, overlap):
    """Step 2: cut text into overlapping pieces."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def main():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    index = pc.Index(PINECONE_INDEX_NAME)

    # wipe old data (default namespace AND the bad "code-helper-self" one)
    for ns in ("", "code-helper-self"):
        try:
            index.delete(delete_all=True, namespace=ns)
        except Exception:
            pass

    # Step 1: READ + Step 2: SPLIT (only the 5 project files, no subfolders)
    records = []
    for path in sorted(PROJECT_PATH.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            records.append({"id": f"{path.name}-{i}", "file": path.name,
                            "chunk_no": i, "text": chunk})
    print(f"Read files from {PROJECT_PATH}")
    print(f"Made {len(records)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Step 3: EMBED + Step 4: STORE
    for b in range(0, len(records), 16):
        batch = records[b:b + 16]
        vectors = embed_texts([r["text"] for r in batch])
        index.upsert(vectors=[
            {"id": r["id"], "values": v,
             "metadata": {"file": r["file"], "chunk_no": r["chunk_no"], "text": r["text"]}}
            for r, v in zip(batch, vectors)
        ])
        print(f"  stored {min(b + 16, len(records))}/{len(records)}")

    print("Done. Pinecone now holds your code.")


if __name__ == "__main__":
    main()
