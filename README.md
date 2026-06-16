# ArXiv RAG

<div align="center">

**Ask questions across multiple academic papers. Get grounded answers with source citations.**

![Stack](https://img.shields.io/badge/Python-3.14-blue?style=flat-square) ![Stack](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square) ![Stack](https://img.shields.io/badge/pgvector-PostgreSQL-336791?style=flat-square) ![Stack](https://img.shields.io/badge/Groq-LLaMA_3.1-orange?style=flat-square) ![Stack](https://img.shields.io/badge/React-TypeScript-61dafb?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

![demo](docs/demo.gif)

</div>

---

## What it does

Instead of reading a 30-page paper to find a specific concept, you upload it and ask directly. The system retrieves the most relevant passages and generates a grounded answer citing which paper the information comes from.

- **Answers stream in token by token**, like ChatGPT or Gemini, so you start reading immediately instead of waiting for the full response to be generated.
- **It remembers the conversation**: follow-up questions like *"how do they apply that?"* reuse the previous messages as context, so you can have a real back-and-forth instead of one-shot questions.
- **You control what's in memory**: upload new PDFs and remove existing ones from the sidebar at any time. Every question is answered across all indexed documents simultaneously.

**Try it with a ready-made set of papers.** Download these foundational ML / information-retrieval papers into `docs/papers/`, then upload them through the UI to start asking questions. `wget -O <destination> <url>` saves each PDF under that name (no `wget`? use `curl -L -o <destination> <url>`, or install it with `sudo apt install wget` / `brew install wget`):

```bash
mkdir -p docs/papers
wget -O docs/papers/01_monobert_reranking_2019.pdf    https://arxiv.org/pdf/1901.04085
wget -O docs/papers/02_dpr_dense_retrieval_2020.pdf   https://arxiv.org/pdf/2004.04906
wget -O docs/papers/03_sentence_bert_2019.pdf         https://arxiv.org/pdf/1908.10084
wget -O docs/papers/04_text_ranking_survey_2021.pdf   https://arxiv.org/pdf/2010.06467
wget -O docs/papers/05_rag_original_2020.pdf          https://arxiv.org/pdf/2005.11401
wget -O docs/papers/06_ragas_2023.pdf                 https://arxiv.org/pdf/2309.15217
wget -O docs/papers/07_hnsw_ann_2016.pdf              https://arxiv.org/pdf/1603.09320
```

These are the same papers the retrieval pipeline itself is built on. See [Further reading](#further-reading) for what each one explains.

---

## Architecture

```
User Question 
      ↓
Query Rewriting (Groq)
      ↓
      ├─► Plain text ─────────► BM25 / PostgreSQL FTS ──┐
      │                                                  ├─► RRF Fusion (top-20)
      └─► Embedding ──────────► Vector Search (pgvector)─┘
                                                                    ↓
                                                        Cross-Encoder Reranking (top-5)
                                                                    ↓
                                                                Groq LLM → Answer
```

Most RAG systems stop at the embedding search step, convert the question to a vector, find similar chunks, send them to the LLM. This works reasonably well but has a key weakness: embedding models compress meaning into a fixed-size vector, which loses information. A question about "model training instability" might not match a chunk that discusses "gradient explosion" even though they're the same concept.

This system addresses that with a three-stage retrieval pipeline:

**1. Hybrid retrieval**

The question is searched two ways simultaneously:
- **Keyword search (BM25)**: PostgreSQL matches exact and related terms in the text, the same way a search engine does. Great for specific terminology, model names, author names.
- **Semantic search**: the question is converted to an embedding (a numerical representation of its meaning) and compared against all stored chunk embeddings. Great for conceptual matches even when the wording differs.

Neither approach is strictly better, they catch different things. Running both maximises recall.

**2. Reciprocal Rank Fusion (RRF)**

Each search returns its own ranked list of 20 candidates. RRF merges them by scoring each chunk as `Σ 1/(60 + rank)` across both lists. A chunk that ranks highly in both searches receives a much higher combined score than one that only appears in one. This surfaces the most consistently relevant results regardless of which retrieval method found them.

**3. Cross-encoder reranking**

The top-20 fused candidates are re-scored by a cross-encoder model (`ms-marco-MiniLM-L-12-v2`). Unlike the embedding model, which encodes the question and each chunk independently, the cross-encoder reads them together, letting it reason about how well a specific passage actually answers the specific question. This is slower but significantly more accurate, which is why it is only applied to the small set of candidates that survived the previous stages. The top-5 go to the LLM.

**Query rewriting**

Before retrieval runs, the user's question is rewritten using the conversation history. Follow-up questions like "how do they apply that?" are meaningless without context: the rewriter turns them into self-contained queries that work correctly in semantic search.

---

## Evaluation

Evaluated with [RAGAS](https://github.com/explodinggradients/ragas):

| Pipeline | answer_relevancy |
|---|---|
| Embeddings only (baseline) | 0.60 - 0.70 |
| **Hybrid search + RRF + reranking** | **0.85 – 0.95** |

The score varies slightly between runs depending on the questions used. The improvement over the baseline reflects the gains from combining keyword and semantic search, and from the cross-encoder surfacing genuinely relevant passages instead of just semantically similar ones.

```bash
# Run evaluation (~3 min: sequential to respect Groq free tier TPM limits)
uv run python -m app.evaluate

# Custom questions
uv run python -m app.evaluate "What is YOLO?" "How does attention work?"
```

---

## Getting started

**Prerequisites:** Docker and Docker Compose.

```bash
git clone https://github.com/JoseGarciaMayen/arxiv-rag
cd arxiv-rag
./setup.sh
```

`setup.sh` will ask for your Groq API key (free at [console.groq.com](https://console.groq.com)), build the images, and start everything.

| URL | What |
|---|---|
| http://localhost:3000 | App |
| http://localhost:8000/docs | Swagger UI (API reference) |

```bash
docker compose down   # stop
docker compose up     # start again (no rebuild)
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and index a PDF |
| `POST` | `/query` | Ask a question (supports conversation history) |
| `POST` | `/query/stream` | Same as `/query`, but streams the answer token by token (SSE) |
| `GET` | `/documents` | List indexed documents with chunk counts |
| `DELETE` | `/document/{filename}` | Remove a document and its embeddings |
| `GET` | `/health` | Health check |

---

## Stack

| Component | Choice | Why |
|---|---|---|
| Vector store | pgvector | FAISS doesn't persist between restarts and doesn't scale to multiple users. Pinecone is a paid external service. pgvector runs inside PostgreSQL, which is already needed for metadata: no extra infrastructure, and the query/hybrid search stays in one place |
| Embeddings | all-MiniLM-L6-v2 | Free, runs fully local, 384 dims sufficient for academic text. `allenai-specter` would be more domain-specific but requires significantly more resources |
| Reranker | ms-marco-MiniLM-L-12-v2 | Cross-encoder trained on MS MARCO passage ranking, runs locally, meaningfully improves relevance over cosine similarity alone |
| LLM | Groq + LLaMA 3.1 8B | Free tier with generous limits, very fast inference. The prompt is decoupled from the model: easy to swap |
| PDF parsing | PyMuPDF | More reliable than pypdf for academic papers with columns, figures, and tables; pypdf was previously used and it fails on some complex layouts |
| Backend | FastAPI | Async, typed, auto-generates Swagger UI at `/docs` |
| Frontend | React + Vite + TypeScript + Tailwind | Served via nginx in Docker: nginx also proxies `/api/*` to FastAPI since the browser can't reach the internal Docker network directly |

---

## Known limitations

- Metadata chunks (acknowledgements, copyright notices) occasionally pass the quality filters and appear in retrieval results
- No authentication on the API: anyone who can reach the server can upload documents or query
- The LLM occasionally uses its own training knowledge even when the prompt instructs it to answer only from the provided context

---

## Further reading

The retrieval pipeline is built on a handful of foundational papers. Each row links to the paper and maps it to the part of the system it explains:

| Paper | Explains |
|---|---|
| [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) (Cormack et al., 2009) | How dense + keyword results are fused (`reciprocal_rank_fusion`) |
| [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) (Nogueira & Cho, 2019) | The cross-encoder reranking stage |
| [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020) | Semantic search and why it complements BM25 |
| [Sentence-BERT](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019) | The `all-MiniLM-L6-v2` embedding model |
| [Text Ranking: BERT and Beyond](https://arxiv.org/abs/2010.06467) (Lin et al., 2021) | Survey tying the whole retrieval stack together |
| [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020) | The original RAG pattern |
| [RAGAS](https://arxiv.org/abs/2309.15217) (Es et al., 2023) | The evaluation metrics in `app/evaluate.py` |
| [HNSW](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2016) | The approximate nearest-neighbour index on embeddings |

These are the same PDFs downloaded by the [`wget` commands in **What it does**](#what-it-does), so once you've run those, you already have them locally both as test documents for the app and as reading material.

> RRF (Cormack 2009) is behind the ACM paywall, so it has no free PDF link. The
> DOI above points to it. BM25 is best read in *Introduction to Information
> Retrieval* (Manning et al., free online), chapter on ranked retrieval.

---

## Project structure

```
arxiv-rag/
├── app/
│   ├── api.py          # FastAPI endpoints
│   ├── ingest.py       # PDF extraction, chunking, embedding, pgvector storage
│   ├── query.py        # Hybrid search, RRF, reranking, LLM generation
│   ├── evaluate.py     # RAGAS evaluation
│   └── models.py       # Shared model instances (embedder + reranker)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatWindow.tsx
│   │   ├── hooks/
│   │   │   └── useChat.ts
│   │   ├── App.tsx
│   │   └── types.ts
│   ├── nginx.conf
│   ├── Dockerfile
│   └── package.json
├── docs/               # Sample papers (gitignored) + demo
│   ├── attention_is_all_you_need.pdf
│   ├── bert.pdf
│   ├── yolo.pdf
│   ├── rag.pdf
│   ├── adam_optimizer.pdf
│   └── demo.gif
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_ingest.py
│   └── test_query.py
├── .github/workflows/ci.yml
├── setup.sh
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```
