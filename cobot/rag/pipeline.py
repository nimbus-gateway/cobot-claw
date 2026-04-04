"""
RAG pipeline — Phase 6 implementation.

Uses LlamaIndex for document loading / chunking / embedding and ChromaDB
as the persistent vector store.

Install extras before using:
    pip install -e ".[rag]"

Index your manuals once:
    python -m cobot.rag.ingest --path ./manuals
    # or via CLI:
    cobot-claw ingest --path ./manuals
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cobot.config import CobotConfig

logger = logging.getLogger(__name__)


def _check_rag_deps() -> bool:
    try:
        import llama_index.core  # noqa: F401
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False


def _configure_embed_model() -> None:
    """
    Explicitly set a local HuggingFace embedding model so LlamaIndex never
    tries to fall back to the OpenAI embeddings package.

    BAAI/bge-small-en-v1.5 is ~130 MB, downloaded once to the HuggingFace
    cache (~/.cache/huggingface).  No API key required.
    """
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def _configure_llm(cfg: "CobotConfig") -> None:
    """
    Explicitly set the LlamaIndex LLM so it never falls back to OpenAI.
    Mirrors the provider selection in cobot/llm.py.
    """
    from llama_index.core import Settings
    provider = cfg.llm.provider
    api_key = cfg.llm.api_key or None  # None → use env var

    if provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        model = cfg.agents.config.model
        Settings.llm = Anthropic(model=model, api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
    elif provider == "openai":
        from llama_index.llms.openai import OpenAI
        model = cfg.agents.config.model
        Settings.llm = OpenAI(model=model, api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    else:
        # local / Ollama — OpenAI-compatible endpoint
        from llama_index.llms.openai_like import OpenAILike
        model = cfg.agents.config.model
        Settings.llm = OpenAILike(
            model=model,
            api_base=cfg.llm.base_url,
            api_key=api_key or "ollama",
            is_chat_model=True,
        )


class RAGPipeline:
    """
    ChromaDB-backed RAG pipeline for robot manual Q&A.

    If the [rag] extras are not installed or the collection is empty the
    pipeline falls back to a helpful message — it never crashes the CLI.
    """

    COLLECTION = "robot_manuals"

    def __init__(self, db_path: str = "./chroma_db", cfg: "CobotConfig | None" = None) -> None:
        self._db_path = str(db_path)
        self._cfg = cfg
        self._index = None  # lazily initialised on first query/ingest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self):
        """Load (or create) the ChromaDB-backed vector index."""
        if self._index is not None:
            return self._index

        if not _check_rag_deps():
            return None

        _configure_embed_model()
        if self._cfg is not None:
            _configure_llm(self._cfg)

        import chromadb
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore

        client = chromadb.PersistentClient(path=self._db_path)
        collection = client.get_or_create_collection(self.COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        return self._index

    def _collection_size(self) -> int:
        """Return number of documents indexed (0 if unavailable)."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self._db_path)
            col = client.get_or_create_collection(self.COLLECTION)
            return col.count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query(self, question: str) -> str:
        """Answer a question using the indexed robot manuals."""
        if not _check_rag_deps():
            return (
                "RAG not available — install the [rag] extras:\n"
                "    pip install -e \".[rag]\"\n"
                "Then index your manuals:\n"
                "    cobot-claw ingest --path ./manuals"
            )

        index = self._load_index()
        if index is None:
            return "Could not load RAG index."

        if self._collection_size() == 0:
            return (
                "No manuals have been indexed yet. Run:\n"
                "    cobot-claw ingest --path ./manuals"
            )

        try:
            query_engine = index.as_query_engine(similarity_top_k=4)
            # LlamaIndex query engines are synchronous — run in executor
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: query_engine.query(question)
            )
            return str(response)
        except Exception as e:
            logger.exception("rag.query_error")
            return f"RAG query failed: {e}"

    async def ingest(self, path: str) -> dict:
        """
        Index all PDF/TXT/MD files found under *path* into ChromaDB.

        Returns a summary dict: {ok, indexed, skipped, errors}.
        """
        if not _check_rag_deps():
            return {
                "ok": False,
                "message": (
                    "RAG extras not installed. Run:\n"
                    "    pip install -e \".[rag]\""
                ),
            }

        source = Path(path)
        if not source.exists():
            return {"ok": False, "message": f"Path not found: {path}"}

        files = (
            list(source.rglob("*.pdf"))
            + list(source.rglob("*.txt"))
            + list(source.rglob("*.md"))
        )
        if not files:
            return {"ok": False, "message": f"No PDF/TXT/MD files found under {path}"}

        try:
            _configure_embed_model()
            if self._cfg is not None:
                _configure_llm(self._cfg)

            import chromadb
            from llama_index.core import (
                VectorStoreIndex,
                StorageContext,
                SimpleDirectoryReader,
            )
            from llama_index.vector_stores.chroma import ChromaVectorStore

            client = chromadb.PersistentClient(path=self._db_path)
            collection = client.get_or_create_collection(self.COLLECTION)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            reader = SimpleDirectoryReader(
                input_files=[str(f) for f in files],
                filename_as_id=True,
            )
            documents = reader.load_data()

            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
            )

            # Invalidate cached index so next query picks up new docs
            self._index = None

            return {
                "ok": True,
                "indexed": len(documents),
                "skipped": 0,
                "collection": self.COLLECTION,
                "db_path": self._db_path,
            }

        except Exception as e:
            logger.exception("rag.ingest_error")
            return {"ok": False, "message": str(e)}
