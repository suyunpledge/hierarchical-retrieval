# hierarchical-retrieval

> A developer toolkit for hierarchical retrieval. It uses three tiers—full-text cache, key-sentence library, and topic domains—on top of local Ollama embeddings (default: `bge-m3`, 1024 dimensions) to expand retrievable history while keeping prompt size bounded. [中文文档](README.zh.md)

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)]()

---

## Table of Contents

- [Background & Motivation](#background--motivation)
- [Core Idea: Hierarchical Retrieval](#core-idea-hierarchical-retrieval)
- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Parameters](#configuration-parameters)
- [API Reference](#api-reference)
- [Retrieval Flow in Detail](#retrieval-flow-in-detail)
- [Real-World Results](#real-world-results)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Security Notes](#security-notes)
- [FAQ](#faq)
- [License](#license)

---

## Background & Motivation

Modern LLMs, including models with very large context windows, have improved long-context handling substantially. They still face several fundamental memory and retrieval problems:

| Problem | Symptom |
|------|------|
| **Full-text-everything leads to forgetting** | Information is lost once the context limit is exceeded |
| **Token waste** | Simple analysis and repeated Q&A consume large amounts of memory tokens |
| **Detail loss & recall difficulty** | Long-text memory suffers from information decay |
| **Topic interference** | Unrelated topics get mixed in, confusing recall |

**Hierarchical retrieval** stores information in tiers and scopes retrieval by topic. The prompt budget is therefore spent on key sentences relevant to the active domain, extending retrievable history without modifying the underlying model.

> Quantification note (as of v0.3): whether hierarchical retrieval actually helps depends on the **retrieval hit rate**, and that needs to be benchmarked against real corpora before quoting a number. The docs no longer cite unverified multiplier estimates (v0.2 once claimed "1M can be used as 100M," which was unfounded and has since been removed). The one hard, verifiable metric is that the `/v1` chat-proxy prompt size is **structurally bounded** — header prompt + memory block (≤ `context_char_budget`) + last K messages (also capped into the same budget, with any single overlong message truncated on assembly) — independent of history length.

---

## Core Idea: Hierarchical Retrieval

Hierarchical retrieval consists of three parts:

### 1. Full-Text Cache (base information store, local disk)
**Caches every piece of conversation, verbatim, to local disk** (`CloudCache`, class name kept for historical reasons), serving as the base information store. In the retrieval pipeline it is the **detail layer**: hit key sentences are linked back to their original passages via `source_entry_id`; it can also be queried directly for full-corpus semantic search via `cache.search()`. An "L1 fallback recall" in the automatic retrieval pipeline is on the roadmap (see the leftover item in the v0.3 fix notes).

### 2. Key-Sentence Library (secondary information store)
During conversation, the AI **distills content into concise key sentences**, stored in a separate space. At retrieval time, key sentences are hit first, then the corresponding detailed conversation is looked up in the full-text cache based on them, avoiding a full-text scan.

### 3. Topic Domains
The AI **partitions conversation content into distinct topic domains**, assigning a key-sentence library to each domain. At retrieval time, the topic domain the query belongs to is determined first, and then precise linking is done against that domain's key-sentence library and the full-text cache.

This way, "context length" only comes into play within the key-sentence library of the **relevant topic domain** — unrelated information is naturally isolated, enabling large-capacity information memory.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Query                                 │
└──────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  L3  Topic Domains (TopicDomainManager)                     │
│  ── Detects which topic the query belongs to, narrows scope │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │ Domain A   │ │ Domain B   │ │ Domain C   │  ...         │
│  └─────┬──────┘ └────────────┘ └────────────┘              │
│        │ Hit domain                                          │
└────────┼────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  L2  Key-Sentence Library (KeySentenceLibrary)               │
│  ── Retrieves most relevant key sentences within the domain  │
│  ┌──────────────────────────────────────────┐               │
│  │ "Adopted microservices architecture..." (score 0.74)     │
│  │ "Database uses PostgreSQL..."  (score 0.68)               │
│  │ "API Gateway routing..."  (score 0.61)                    │
│  └──────────────────┬───────────────────────┘               │
│                     │ Key sentence links source_entry_id     │
└─────────────────────┼───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  L1  Full-Text Cache (CloudCache, local disk)                │
│  ── Links back to full-text passages via key sentences       │
│  ┌──────────────────────────────────────────┐               │
│  │ entry_id: cc_xxx  chunk: 0  full text...  │               │
│  │ entry_id: cc_yyy  chunk: 1  full text...  │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Aggregated result: { topic domain, key sentences, full-text │
│  passages, summary }                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

- **Three-tier hierarchical retrieval** — full-text cache → key-sentence library → topic domains, progressively narrowing the retrieval scope
- **bge-m3 integration** — called via the Ollama API, vectors are automatically L2-normalized (dot product = cosine similarity)
- **FAISS vector index** — based on `IndexFlatIP`, supports persistence to disk, auto-restored on process restart
- **Automatic topic clustering** — new content is automatically assigned to an existing topic or a new one is created, with topic vectors updated smoothly
- **Heuristic key-sentence extraction** — lightweight extraction based on signal-word weight, position, and length; no extra LLM call needed
- **Fallback retrieval** — when no topic matches, automatically falls back to global key-sentence search to help close recall gaps
- **Full persistence** — all data is written to local disk; zero external service dependency (aside from Ollama)
- **Configurable thresholds** — topic merge/match thresholds, top-k, and other parameters are all tunable

---

## Installation

### Prerequisites

| Step | Description | Command |
|------|------|------|
| 1. Python | 3.9+ required | `python --version` |
| 2. Pull the model | Download the bge-m3 embedding model (default) | `ollama pull bge-m3` |
| 3. Start Ollama | Listens on `http://localhost:11434` by default | `ollama serve` |

### Install this package

| Method | Description | Command |
|------|------|------|
| Dev mode | Editable source install (recommended for development) | `pip install -e .` |
| Regular install | Build then install into site-packages | `pip install .` |
| GPU acceleration | Use `faiss-gpu` instead of `faiss-cpu` | `pip install -e ".[gpu]"` |
| API service | Includes FastAPI / uvicorn / pydantic | `pip install -e ".[api]"` |
| Dev dependencies | Includes pytest / build / twine | `pip install -e ".[dev]"` |

### Runtime dependencies

| Dependency | Purpose |
|------|------|
| `numpy` | Vector math |
| `requests` | Calling the Ollama API |
| `faiss-cpu` | Approximate-nearest-neighbor vector indexing |

---

## Quick Start

```python
from hierarchical_retrieval import HConfig, HierarchicalRetrieval

# 1. Initialize (auto-connects to Ollama)
hr = HierarchicalRetrieval(config=HConfig(storage_root="./my_store"))

# 2. Ingest a conversation (automatic: full-text cache + key-sentence extraction + topic assignment)
hr.ingest(
    "Discussed the technical architecture today. Recommended microservices, "
    "database on PostgreSQL. Final decision: start building the framework next week.",
    metadata={"source": "arch_meeting"},
)

# 3. Retrieve
result = hr.retrieve("What architecture approach was decided?")
print(result["summary"])
print(result["key_sentences"])    # list of key sentences
print(result["full_contexts"])    # linked full-text passages
```

### Run the example

```bash
python hierarchical_retrieval/examples/demo.py
```

---

## Configuration Parameters

All `HConfig` parameters, grouped by function. The third column shows recommended values for Chinese-language scenarios (defaults are tuned for English; Chinese usage needs a higher topic threshold).

**Embedding service**

| Parameter | Default | Chinese recommendation | Description |
|------|--------|----------|------|
| `embedding_model` | `bge-m3:latest` | same as default | Ollama embedding model name |
| `embedding_dim` | `1024` | same as default | Embedding vector dimension |
| `embedding_base_url` | `http://localhost:11434` | same as default | Ollama service address |
| `embedding_request_timeout` | `60` | same as default | Request timeout (seconds) |

**Storage paths**

| Parameter | Default | Description |
|------|--------|------|
| `storage_root` | `./hierarchical_storage` | Storage root directory |
| `cloud_cache_dir` | `cloud_cache` | Full-text cache subdirectory |
| `key_sentence_dir` | `key_sentence_lib` | Key-sentence library subdirectory |
| `topic_domain_dir` | `topic_domains` | Topic domain subdirectory |
| `vector_index_dir` | `vector_index` | Vector index subdirectory |

**Retrieval parameters**

| Parameter | Default | Chinese recommendation | Description |
|------|--------|----------|------|
| `top_k_key_sentences` | `5` | `3` | Number of key sentences returned per retrieval |
| `top_k_full_context` | `3` | `2` | Number of full-text passages linked per key sentence |
| `key_sentence_similarity_threshold` | `0.55` | same as default | Key-sentence match similarity threshold (empirically chosen from bge-m3 distributions) |

**Topic domains**

| Parameter | Default | Chinese recommendation | Description |
|------|--------|----------|------|
| `topic_similarity_threshold` | `0.55` | same as default | Similarity threshold for matching a query to a topic (empirically chosen from bge-m3 distributions) |
| `topic_min_similarity` | `0.62` | same as default | Minimum similarity to assign content into an existing topic on write (empirically chosen from bge-m3 distributions) |
| `topic_max_domains` | `100` | same as default | Maximum number of topic domains |

**Key-sentence extraction**

| Parameter | Default | Description |
|------|--------|------|
| `max_key_sentences_per_chunk` | `3` | Maximum key sentences extracted per chunk |
| `key_sentence_max_length` | `128` | Maximum character length of a single key sentence |

**Context assembly & proxy (used by the `/v1` chat proxy)**

| Parameter | Default | Description |
|------|--------|------|
| `context_char_budget` | `3000` | Character budget for the memory block / recent conversation (lowest-scoring items dropped over budget; any single overlong message is truncated on assembly) |
| `recent_turns_keep` | `6` | Number of recent **messages** kept verbatim (roughly 3 turns of conversation) |
| `compress_threshold_chars` | `3000` | Proxy compression triggers only once total history characters exceed this value |
| `chat_backend_url` | `http://localhost:11434` | Proxy forwarding target (local Ollama) |
| `chat_model` | `""` (empty) | Fallback chat model for the proxy; must be set via the `HR_CHAT_MODEL` environment variable or code |

**Performance & proxy protection**

| Parameter | Default | Description |
|------|--------|------|
| `embed_workers` | `6` | Number of concurrent threads for batch embedding (Ollama supports concurrency) |
| `max_ingest_per_request` | `32` | Maximum number of older messages the proxy will backfill-ingest per single request (not-yet-ingested ∩ oldest-first) |

**Other**

| Parameter | Default | Description |
|------|--------|------|
| `log_level` | `INFO` | Log level |

> **Chinese-language tuning tip**: the default embedding is bge-m3 (1024-dim), and the default thresholds (0.55 / 0.55 / 0.62) were chosen from empirical measurements on Chinese-language distributions, so they usually work out of the box. If your corpus still shows mis-merged topics or missed recalls, fine-tune these two thresholds. Feel free to copy this directly:

```python
config = HConfig(
    storage_root="./my_store",
    topic_min_similarity=0.62,       # write-time merge threshold (empirically chosen for bge-m3)
    topic_similarity_threshold=0.55, # query-match threshold (empirically chosen for bge-m3)
    top_k_key_sentences=3,
    top_k_full_context=2,
)
```

---

## API Reference

### `HierarchicalRetrieval` — main entry point

```python
hr = HierarchicalRetrieval(embedding=None, config=None)
```

| Method | Description |
|------|------|
| `ingest(text, metadata=None) -> dict` | Ingests a piece of conversation into the three-tier storage, returns `{entry_id, domain_id, domain_name, sentence_ids}` |
| `ingest_batch(conversations) -> list[dict]` | Batch ingest |
| `retrieve(query, top_k_sentences=None, top_k_context=None) -> dict` | Hierarchical retrieval, returns `{query, detected_domain, key_sentences, full_contexts, summary}` |
| `status() -> dict` | Returns system status (counts at each tier, list of topics) |

### `NomicEmbedding` — embedding service

```python
emb = NomicEmbedding(config=None)
emb.embed("some text")        # -> np.ndarray (1024,)
emb.embed_batch(["t1"])       # -> list[np.ndarray]
```

### `CloudCache` — L1 full-text cache

```python
cache = CloudCache(embedding, config=None)
cache.store(text, metadata=None)   # -> entry_id
cache.search(query, top_k=3)       # -> list[dict]
cache.has_content(text)            # -> bool (content-hash-based dedup lookup)
cache.get_full_text(entry_id)      # -> str | None
cache.count()                      # -> int
```

### `KeySentenceLibrary` — L2 key-sentence library

```python
ks_lib = KeySentenceLibrary(embedding, config=None)
ks_lib.ingest(text, source_entry_id)  # -> list[dict] (includes sentence_id / text / vector, reusable by domain indexing)
ks_lib.search(query, top_k=5)         # -> list[dict]
ks_lib.get_sentence(sentence_id)      # -> dict | None
```

### `TopicDomainManager` — L3 topic domains

```python
tm = TopicDomainManager(embedding, config=None)
tm.assign_domain(text)              # -> TopicDomain (assigns to existing or creates new)
tm.detect_domain(query)             # -> TopicDomain | None
tm.search_in_domain(query, domain)  # -> list[dict]
tm.list_domains()                   # -> list[TopicDomain]
```

### Storage and utility classes

- `LocalStorage` — local key-value store based on JSON/Pickle
- `VectorStore` — FAISS vector index wrapper, supports persistence
- `TextProcessor` — sentence splitting, chunking, cleaning
- `KeySentenceExtractor` — heuristic key-sentence extraction

---

## Retrieval Flow in Detail

```
Input: query
│
├─ Step 1: Topic detection (L3)
│    Vectorize the query, compute cosine similarity against every topic domain's representative vector
│    ├─ Hit (≥ topic_similarity_threshold) → enter that domain
│    └─ No hit → fall back to global key-sentence search
│
├─ Step 2: In-domain key-sentence retrieval (L2)
│    Retrieve top_k_key_sentences key sentences in the hit domain's independent vector space
│    First filter out low-score noise via key_sentence_similarity_threshold;
│    if fewer than top_k remain after filtering, supplement with global key-sentence search (same filtering) and deduplicate
│
├─ Step 3: Key sentence → full-text linking (L1)
│    For each key sentence's source_entry_id, retrieve the full-text passage from the full-text cache
│    Deduplicate and take up to top_k_full_context * top_k_key_sentences passages
│
└─ Step 4: Aggregate
     Generate the summary text and return the full result dict
```

---

## Real-World Results

Using 4 conversations across different topics (technical architecture / budget / launch plan / architecture follow-up) as an example:

**Ingestion result** — automatically formed 3 topic domains:
- Technical architecture domain (2 entries; the architecture plan and its follow-up were auto-merged)
- Budget domain (1 entry)
- Launch-plan domain (1 entry)

**Retrieval results**:

| Query | Matched topic | Top-1 key sentence |
|------|----------|-------------|
| Technical architecture plan / database choice | Technical architecture domain | "I'll add some detail to the architecture plan we just discussed" |
| Q3 budget / GPU compute | Budget domain | "About the budget — the Q3 R&D budget needs to be re-planned" |
| Launch timing / security audit | Launch-plan domain | "Suggest assigning someone to track progress on each milestone" |
| API Gateway choice | Technical architecture domain | "For API Gateway, we'll go with Kong or APISIX..." |

Every query was correctly routed to the matching topic domain, key sentences hit precisely, and full-text passages linked successfully.

---

## Project Structure

```
hierarchical-retrieval/
├── pyproject.toml              # Packaging config (PEP 517/518)
├── README.md                   # This document
├── LICENSE                      # MIT
├── .gitignore
│
└── hierarchical_retrieval/     # Main package
    ├── __init__.py             # Public API exports
    ├── config.py               # HConfig global configuration
    ├── py.typed                # PEP 561 type marker
    │
    ├── core/                   # Core engine
    │   ├── embedding.py        # NomicEmbedding — Ollama embedding client
    │   ├── cloud_cache.py      # CloudCache — L1 full-text cache
    │   ├── key_sentence.py     # KeySentenceLibrary — L2 key-sentence library
    │   ├── topic_domain.py     # TopicDomainManager — L3 topic domains
    │   └── retrieval.py        # HierarchicalRetrieval — three-tier pipeline
    │
    ├── storage/                # Storage layer
    │   ├── base.py             # StorageBase abstract interface
    │   ├── local_storage.py    # LocalStorage local key-value store
    │   └── vector_store.py     # VectorStore FAISS index
    │
    ├── utils/                  # Utilities
    │   ├── text_processor.py   # Text splitting/cleaning
    │   └── summarizer.py       # Heuristic key-sentence extraction
    │
    └── examples/
        └── demo.py             # Complete usage example
```

---

## Advanced Usage

### Custom embedding service

```python
from hierarchical_retrieval import HConfig

# Use a different Ollama embedding model
config = HConfig(
    embedding_model="mxbai-embed-large:latest",
    embedding_dim=1024,
)
```

### Working directly with each tier

```python
from hierarchical_retrieval import HConfig, NomicEmbedding, CloudCache

emb = NomicEmbedding(HConfig())
cache = CloudCache(emb)

# Manually store the full text
entry_id = cache.store("long conversation full text...", metadata={"date": "2026-08-06"})

# Retrieve on demand
results = cache.search("related question", top_k=5)
```

### Batch ingestion

```python
conversations = [
    {"text": "conversation 1...", "metadata": {"src": "a"}},
    {"text": "conversation 2...", "metadata": {"src": "b"}},
]
hr.ingest_batch(conversations)
```

### Checking system status

```python
status = hr.status()
# {
#     "embedding_model": "bge-m3:latest",
#     "cloud_cache_entries": 4,
#     "key_sentence_count": 12,
#     "topic_domains": 3,
#     "domains": [{"id": "td_xxx", "name": "...", "entries": 2}, ...]
# }
```

---

## Security Notes

The `/v1` chat proxy receives and caches full conversation content; the default deployment shape is designed for "local-only use" (aligned with the implementation as of v0.3):

- **Local-only listening by default**: all three startup methods (`start_api.bat`, `start_hr_api_hidden.vbs`, and the `hr-api-task.xml` scheduled task) default `--host` to `127.0.0.1` and no longer bind to `0.0.0.0`;
- **Optional Bearer auth**: setting the `HR_API_TOKEN` environment variable makes every endpoint except `/health` require an `Authorization: Bearer <your token>` header. For LobeChat integration, just fill in the same value as the "API Key"; if unset, no auth is enforced (acceptable risk when listening locally only);
- **For LAN/remote access**: change the `--host` startup argument to `0.0.0.0`, **and be sure to also set `HR_API_TOKEN`** — the proxy forwards the full plaintext conversation and should not be exposed to a non-local network without authentication;
- **CORS**: `allow_origins=["*"]` is kept but without credentials (as of v0.3, the non-compliant `allow_credentials` combination has been removed);
- **Data-at-rest location**: all data (full text, key sentences, vector indexes) is written to local disk under the `storage_root` directory and is never uploaded to any remote service.

---

## FAQ

**Q: On startup I get "unable to connect to the Ollama service"?**

A: Make sure `ollama serve` is running and reachable at `http://localhost:11434` (the default address). If Ollama is running elsewhere, configure it via `HConfig(embedding_base_url="http://your-host:11434")`.

**Q: All conversations get assigned to the same topic?**

A: This is usually caused by a too-low similarity threshold. The default (`topic_min_similarity` = 0.62) was chosen based on bge-m3's Chinese-language distribution; raise it further if mis-merging still occurs.

**Q: Key sentences come back empty on retrieval?**

A: Check whether data has actually been ingested (`hr.status()`), and confirm the query is semantically related to what's stored. When no topic matches, retrieval automatically falls back to a global search.

**Q: How do I reset all data?**

A: Delete the `storage_root` directory (default `./hierarchical_storage`, or `./demo_storage` for the demo).

**Q: Is GPU acceleration supported?**

A: Yes. Install `faiss-gpu` instead of `faiss-cpu`: `pip install -e ".[gpu]"`.

---

## License

[MIT](LICENSE)
