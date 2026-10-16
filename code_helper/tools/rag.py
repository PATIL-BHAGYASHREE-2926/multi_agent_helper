from pinecone import Pinecone
from ..settings import PINECONE_API_KEY, PINECONE_INDEX_NAME
from .embeddings import embed_texts

_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)


def rag_search(question: str) -> str:
    """Search the code project and return the code pieces closest to the question.
    Use this to find WHERE something is done (search, embedding, config, etc.)."""
    vector = embed_texts([question])[0]
    result = _index.query(vector=vector, top_k=6, include_metadata=True)
    if not result.matches:
        return "I could not find that."
    parts = []
    for m in result.matches:
        parts.append(
            f"[{m.metadata['file']} | piece {m.metadata['chunk_no']} | score {m.score:.2f}]\n"
            f"{m.metadata['text']}"
        )
    return "\n\n---\n\n".join(parts)
