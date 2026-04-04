"""
RAG pipeline stub for Phase 4/5.

Phase 6 replaces this with the real ChromaDB + LlamaIndex implementation.
Install `pip install -e ".[rag]"` and run `python -m cobot.rag.ingest` before
upgrading to the real pipeline.
"""

from __future__ import annotations


class RAGPipeline:
    """No-op stub — returns a helpful message until Phase 6 is installed."""

    def __init__(self, db_path: str = "./chroma_db") -> None:
        pass

    async def query(self, question: str) -> str:
        return "RAG not available — install [rag] extras and index manuals first."

    async def ingest(self, path: str) -> dict:
        return {"ok": False, "message": "RAG not available."}
