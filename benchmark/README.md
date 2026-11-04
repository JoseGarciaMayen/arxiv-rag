# Retrieval benchmark data

`corpus.jsonl` is generated from the PDFs in `docs/` by running
`uv run python scripts/build_retrieval_corpus.py`. Extraction, cleaning,
chunking (1500 characters with 150 overlap), and filtering reuse the ingestion
pipeline. `corpus-manifest.json` records the input PDF hashes, preprocessing
settings, and corpus hash.

Each corpus row has a stable `chunk_id` composed of the PDF filename and the
SHA-256 hash of the exact chunk text. The ordinal is informational only and is
not part of the identifier. Do not use database serial IDs for benchmark labels.

`queries.jsonl` contains manually authored questions and manually selected
relevant chunk IDs. The labels are binary: a chunk is relevant if it contains
evidence that directly answers the question. These annotations must be reviewed
by a human when changed; do not generate them automatically.

Start a dedicated local PostgreSQL/pgvector service (the separate Compose
project name gives it its own persistent volume):

```bash
sudo docker compose -p arxiv-rag-benchmark up -d db
uv run python scripts/load_retrieval_benchmark_corpus.py \
  --database-url postgresql://rag:rag@localhost:5432/arxiv \
```

This loads the frozen corpus into an isolated `benchmark_chunks` table; it
doesn't delete or modify the app's `chunks` table. Then run the comparison:

```bash
uv run python scripts/run_retrieval_benchmark.py \
  --database-url postgresql://rag:rag@localhost:5432/arxiv \
  --output benchmark/results.json
```

The runner compares PostgreSQL FTS (`ts_rank`), dense vector retrieval, and
their RRF hybrid using the same 30 queries. It reports macro Recall@5/10/20,
MRR, and nDCG@5/10/20. Results include the corpus and question-file hashes.
Both scripts refuse non-local database hosts and never load `.env`; no remote
database or LLM API is needed. Dense vectors use the app's `all-MiniLM-L6-v2`
embedding model. Results are generated output and should only be committed
intentionally after reviewing them.

The benchmark corpus includes the local PDFs as-is, including duplicate copies
where present. This preserves a transparent mapping to the repository sources;
deduplication or exclusions should be an explicit, versioned dataset change.
