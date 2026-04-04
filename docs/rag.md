# RAG — Manual Q&A and Teach Mode

This document explains how the RAG (Retrieval-Augmented Generation) system works in cobot-claw, what libraries are involved, and how each piece connects.

---

## What problem does RAG solve?

LLMs like Claude or GPT-4 have general knowledge, but they do not know the specifics of your robot model, firmware version, or the exact error codes in your UR3e manual. Without RAG, asking *"what is error code 53?"* or *"what is the TCP payload limit?"* would produce a hallucinated or generic answer.

RAG fixes this by searching your actual documentation first, then handing the relevant excerpts to the LLM as context. The LLM then answers based on what it read from your manuals — not from memory.

---

## High-level data flow

```
Your PDF / TXT / MD manuals
        │
        ▼
 SimpleDirectoryReader          ← LlamaIndex reads + parses files
        │
        ▼
 Document chunks (512 tokens)   ← text is split into overlapping windows
        │
        ▼
 Embedding model                ← each chunk → a vector (list of numbers)
        │
        ▼
 ChromaDB (persistent on disk)  ← vectors stored in ./chroma_db
```

Then at query time:

```
User question: "what is the max TCP payload?"
        │
        ▼
 Embedding model                ← question → a vector
        │
        ▼
 ChromaDB similarity search     ← find the 4 most similar chunks
        │
        ▼
 LLM (Claude / GPT-4 / Ollama)  ← receives question + retrieved chunks
        │
        ▼
 Grounded answer to user
```

---

## Libraries

| Library | Role |
|---|---|
| **LlamaIndex Core** (`llama-index-core`) | Orchestrates document loading, chunking, embedding, and querying |
| **LlamaIndex File Reader** (`llama-index-readers-file`) | Parses PDFs, plain text, and Markdown into LlamaIndex `Document` objects |
| **LlamaIndex Chroma** (`llama-index-vector-stores-chroma`) | Adapter that connects LlamaIndex's `VectorStoreIndex` to a ChromaDB collection |
| **ChromaDB** (`chromadb`) | Persistent vector database — stores embeddings on disk so you only need to index once |

Install all of them with:

```bash
pip install -e ".[rag]"
```

---

## Code walkthrough

### `cobot/rag/pipeline.py` — `RAGPipeline`

This is the central class. It has two public methods: `ingest` and `query`.

#### `ingest(path)` — building the index

```
cobot-claw ingest --path ./manuals
```

What happens step by step:

1. **Find files** — scans the directory for `*.pdf`, `*.txt`, `*.md` recursively.
2. **Load documents** — `SimpleDirectoryReader` reads each file. For PDFs it extracts text page by page. Each file becomes one or more `Document` objects.
3. **Open ChromaDB** — a `PersistentClient` opens (or creates) the database at `./chroma_db`. The collection is always named `robot_manuals`.
4. **Build the vector store** — `VectorStoreIndex.from_documents()` does three things:
   - Splits each `Document` into overlapping chunks (default 512 tokens, 64 overlap — set in `cobot.yaml`)
   - Embeds each chunk using the default embedding model (OpenAI `text-embedding-ada-002` when using OpenAI, or a local model when using Ollama)
   - Writes the vectors into ChromaDB
5. **Persist** — ChromaDB automatically saves to disk. The `self._index` cache is invalidated so the next query re-reads the updated collection.

The result is a `{"ok": True, "indexed": N, ...}` dict reported back to the CLI.

#### `query(question)` — answering questions

What happens step by step:

1. **Load index** — `_load_index()` opens ChromaDB and wraps the collection in a `VectorStoreIndex`. This is cached after the first call.
2. **Check collection** — `_collection_size()` returns 0 if nothing has been indexed yet. If empty, returns a helpful message instead of searching.
3. **Create query engine** — `index.as_query_engine(similarity_top_k=4)` builds a retriever that will find the 4 most similar chunks to the question.
4. **Run query** — LlamaIndex embeds the question, queries ChromaDB for the nearest chunks, then calls the LLM with the question and retrieved text as context.
5. **Return answer** — the LLM's response is returned as a string.

Because LlamaIndex's query engine is synchronous, it is wrapped in `loop.run_in_executor(None, ...)` so it does not block the async CLI event loop.

#### Graceful fallback

`_check_rag_deps()` tries to import `llama_index.core` and `chromadb`. If either is missing (i.e. `pip install -e ".[rag]"` was not run), every method returns a helpful message string rather than raising an `ImportError`. The CLI never crashes — it just tells you what to install.

---

### `cobot/rag/ingest.py` — standalone ingest script

A thin CLI wrapper around `RAGPipeline.ingest()`. Can be run two ways:

```bash
# Via entry point
cobot-claw ingest --path ./manuals

# Via module
python -m cobot.rag.ingest --path ./manuals --db ./chroma_db
```

It reads `rag.db_path` from `cobot.yaml` automatically so you do not need to specify `--db` manually. Shows a Rich spinner while indexing.

---

### `cobot/agents/config_agent.py` — Config Agent

The Config Agent is a PydanticAI agent that has access to two tools:

| Tool | What it does |
|---|---|
| `manual_query(question)` | Calls `RAGPipeline.query()` and returns the answer to the LLM |
| `teach_mode(name, duration_sec)` | Records joint states from the arm MCP at 0.5 s intervals, downsamples, and saves a scenario YAML |

When the Orchestrator Agent decides a user question is about configuration or manuals, it delegates to this agent via `run_config_task`.

---

### `cobot/agents/deps.py` — `ConfigAgentDeps`

The Config Agent receives its dependencies through a typed dataclass:

```python
@dataclass
class ConfigAgentDeps:
    arm_mcp: MCPClient          # to read joint states during teach mode
    rag_pipeline: RAGPipeline   # to answer manual questions
    scenarios_store_path: str   # where to save recorded scenarios
    cfg: CobotConfig
```

This is wired in `cobot/cli/app.py` during `_connect_session()` — a `RAGPipeline` instance is created pointing at `cfg.rag.db_path`, and the `ConfigAgentDeps` is attached to `OrchestratorDeps.config_deps`.

---

### `cobot/cli/repl.py` — `/teach` and `/ingest` commands

Two slash commands were added:

**`/teach <name> [--dur N]`**

1. Prints a prompt asking the user to enable freedrive on the pendant and press Enter.
2. Shows a spinner while recording.
3. Calls `build_config_agent(cfg).run(...)` which invokes `teach_mode` internally.
4. The Config Agent calls `get_joint_states` on the arm MCP every 0.5 s, collects waypoints, downsamples (every 3rd point), and writes a YAML file like:

```yaml
name: my_motion
description: Recorded via teach mode (12 waypoints)
robot: default
steps:
  - type: move_joints
    joints: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    duration_sec: 1.5
  - ...
```

The saved file appears immediately in `/scenarios` and can be run with `/run my_motion`.

**`/ingest [path]`**

Calls `session.deps.config_deps.rag_pipeline.ingest(path)` directly (no agent involved) and shows the indexed count.

---

## Configuration

All RAG settings live in `cobot.yaml`:

```yaml
rag:
  manuals_path: ./manuals    # default source path for ingest
  db_path: ./chroma_db       # where ChromaDB stores its files
  chunk_size: 512            # tokens per chunk (not yet wired to LlamaIndex Settings)
  chunk_overlap: 64          # overlap between adjacent chunks
```

`manuals/` and `chroma_db/` are both in `.gitignore` — manuals are large binary files and the vector store is a build artifact that can be regenerated by re-running `ingest`.

---

## What the embedding model is

cobot-claw uses `BAAI/bge-small-en-v1.5` — a local HuggingFace embedding model configured explicitly in `_configure_embed_model()` inside `pipeline.py`. It is set via `Settings.embed_model` before any LlamaIndex indexing or querying happens.

This model:
- Is ~130 MB, downloaded once to `~/.cache/huggingface` on first use
- Runs entirely on your machine — no API key required
- Works regardless of whether your LLM provider is Anthropic, OpenAI, or local Ollama

**Why not use the LlamaIndex default?**

LlamaIndex's default embed model tries to load `llama_index.embeddings.openai`, which requires the `llama-index-embeddings-openai` package and an OpenAI API key. This breaks for Anthropic and local Ollama users. Pinning to HuggingFace makes the embedding step provider-agnostic.

**Switching to a different embedding model**

Edit `_configure_embed_model()` in `cobot/rag/pipeline.py`:

```python
# Larger, more accurate model (~440 MB)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Ollama (if you want embeddings served by Ollama too)
# pip install llama-index-embeddings-ollama
from llama_index.embeddings.ollama import OllamaEmbedding
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
```

If you change the embedding model after indexing, you must re-index — vectors from different models are not compatible.

---

## Sequence: first-time setup to answering a question

```
1. pip install -e ".[rag]"
2. Place manuals in ./manuals/
3. cobot-claw ingest --path ./manuals
   └─ reads PDFs → chunks → embeds → writes ./chroma_db/
4. cobot-claw
5. > what is the default tool speed?
   └─ Orchestrator → run_config_task
      └─ Config Agent → manual_query("what is the default tool speed?")
         └─ RAGPipeline.query(...)
            ├─ embed question
            ├─ ChromaDB: find top-4 similar chunks
            └─ LLM: question + chunks → answer
```

After step 3 the index persists on disk. Step 3 only needs to be repeated when you add new manuals.
