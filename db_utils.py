# db_utils.py
import os, pathlib
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore   # ⬅️ add this

from chromadb import PersistentClient

EMBED_MODEL = "text-embedding-3-small"


def _abs(path: str) -> str:
    """Return absolute path independent of caller’s cwd."""
    return str(pathlib.Path(path).expanduser().resolve())


def get_vectordb():
    backend = os.getenv("VECTOR_BACKEND", "supabase").lower()

    if backend == "chroma":
        persist_dir = _abs(os.getenv("CHROMA_DIR", "./chroma_db"))
        collection  = os.getenv("CHROMA_COLLECTION", "torchlite")

        client = PersistentClient(path=persist_dir)
        return Chroma(
            client=client,
            collection_name=collection,
            embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        )

    # ── default: Supabase ────────────────────────────────────────────────
    supabase: Client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )
    table = os.getenv("SUPABASE_TABLE", "torchlite_embeddings")

    return SupabaseVectorStore(
        client=supabase,
        embedding=OpenAIEmbeddings(model=EMBED_MODEL),
        table_name=table,
        query_name=f"match_{table}",
    )
