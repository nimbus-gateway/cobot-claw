"""
Standalone ingest script — indexes robot manuals into ChromaDB.

Usage:
    python -m cobot.rag.ingest --path ./manuals
    python -m cobot.rag.ingest --path ./manuals --db ./chroma_db --config cobot.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Index robot manuals into the RAG knowledge base."
    )
    p.add_argument(
        "--path",
        default="./manuals",
        help="Directory containing PDF/TXT/MD files (default: ./manuals)",
    )
    p.add_argument(
        "--db",
        default=None,
        help="ChromaDB path (default: taken from cobot.yaml rag.db_path)",
    )
    p.add_argument(
        "--config",
        default="cobot.yaml",
        help="cobot.yaml config file (default: cobot.yaml)",
    )
    return p.parse_args()


async def _run(manuals_path: str, db_path: str) -> None:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from cobot.rag.pipeline import RAGPipeline

    console = Console()
    pipeline = RAGPipeline(db_path=db_path)

    source = Path(manuals_path)
    files = (
        list(source.rglob("*.pdf"))
        + list(source.rglob("*.txt"))
        + list(source.rglob("*.md"))
    )

    if not files:
        console.print(f"[yellow]No PDF/TXT/MD files found under [bold]{manuals_path}[/bold][/yellow]")
        sys.exit(1)

    console.print(f"\n  Found [bold]{len(files)}[/bold] file(s) under [dim]{manuals_path}[/dim]")
    for f in files:
        console.print(f"    [dim]•[/dim] {f.name}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Indexing documents into ChromaDB…", total=None)
        result = await pipeline.ingest(manuals_path)

    if result.get("ok"):
        console.print(
            f"  [green]✓[/green]  Indexed [bold]{result['indexed']}[/bold] document(s) "
            f"into collection [bold]{result['collection']}[/bold]\n"
            f"       DB path: [dim]{result['db_path']}[/dim]\n"
        )
    else:
        console.print(f"  [red]✗[/red]  Ingest failed: {result.get('message')}\n")
        sys.exit(1)


def main() -> None:
    args = _parse_args()

    db_path = args.db
    if db_path is None:
        try:
            from cobot.config import load_config
            cfg = load_config(args.config)
            db_path = cfg.rag.db_path
        except Exception:
            db_path = "./chroma_db"

    asyncio.run(_run(args.path, db_path))


if __name__ == "__main__":
    main()
