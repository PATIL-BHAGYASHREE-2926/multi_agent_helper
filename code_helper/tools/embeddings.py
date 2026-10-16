
from openai import OpenAI
from ..settings import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_EMBED_MODEL,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert texts into embedding vectors using NVIDIA NIM."""
    client = OpenAI(
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
    )

    response = client.embeddings.create(
        input=texts,
        model=NVIDIA_EMBED_MODEL,
    )

    return [item.embedding for item in response.data]
