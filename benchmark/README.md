# Retrieval benchmark data

`corpus.jsonl` is generated from the PDFs in `docs/` by running
`uv run python scripts/build_retrieval_corpus.py`. Extraction, cleaning,
chunking (1500 characters with 150 overlap), and filtering reuse the ingestion
pipeline. `corpus-manifest.json` records the input PDF hashes, preprocessing
settings, and corpus hash.

Each corpus row has a stable `chunk_id` composed of the PDF filename and the
SHA-256 hash of the exact chunk text. Exact duplicate PDFs are excluded; the
lexically first filename is retained and the exclusion is recorded in the
manifest. The ordinal is informational only and is not part of the identifier.
Do not use database serial IDs for benchmark labels.

`queries.jsonl` contains manually authored questions and manually selected
relevant chunk IDs. It covers 50 questions across all 11 unique PDFs. Labels
are binary and may include multiple chunks: a chunk is relevant if it contains
evidence that directly answers some part of the question. Labels must be
reviewed by a human when changed; do not generate them automatically.

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
  --hypothesis "Dense retrieval should outperform FTS on paraphrased questions" \
  --rationale "The pilot set contains conceptual questions whose wording differs from paper passages" \
  --output benchmark/experiments/<unique-run-name>.json
```

The runner compares PostgreSQL FTS (`ts_rank`), local Okapi BM25, dense vector
retrieval, dense+FTS RRF, and dense+BM25 RRF using the same questions. BM25 is
implemented locally over the frozen JSONL corpus with `k1=1.5`, `b=0.75`; it is
not PostgreSQL FTS. It reports macro Recall@5/10/20,
MRR, and nDCG@5/10/20. Results include the corpus and question-file hashes.
Both scripts refuse non-local database hosts and never load `.env`; no remote
database or LLM API is needed. Dense vectors use the app's `all-MiniLM-L6-v2`
embedding model. Results are generated output and should only be committed
intentionally after reviewing them. Save every run under a new filename in
`benchmark/experiments/`; the runner refuses to overwrite a previous run. See
`DECISIONS.md` for the protocol, interpretation limits, and current annotation
status. `REFERENCES.md` summarizes RAG evaluation papers and proposes the next
retrieval, answer-quality, robustness, and operational metrics.

The first run on the pre-deduplication 30-question set is retained as a
historical baseline in `benchmark/experiments/`.
