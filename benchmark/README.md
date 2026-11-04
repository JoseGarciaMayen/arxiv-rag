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

The benchmark corpus includes the local PDFs as-is, including duplicate copies
where present. This preserves a transparent mapping to the repository sources;
deduplication or exclusions should be an explicit, versioned dataset change.
