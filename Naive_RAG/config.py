"""
config.py — Centralized Configuration for Naive RAG Pipeline.

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
