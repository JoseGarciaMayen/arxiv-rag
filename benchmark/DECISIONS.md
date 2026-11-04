# Benchmark protocol and decision log

This is a small **pilot/development benchmark**, not yet a publication-grade
evaluation. Record every experiment as a new JSON file under `experiments/`;
never overwrite a prior run. Each record should include its hypothesis and
rationale, source commit, corpus/question hashes, method configuration, and
per-question ranked chunk IDs and scores.

## Dataset decisions

- Build the corpus only from checked-in local PDFs using the app's current
  extraction, cleaning, chunking, and validity filter. This avoids a mismatch
  between benchmark preprocessing and production preprocessing.
- Exclude byte-identical PDFs by SHA-256; keep the lexically first filename as
  the canonical source and record every exclusion in the corpus manifest. This
  prevents a duplicated paper from being represented twice in retrieval.
- Identify passages by canonical source name plus SHA-256 of exact chunk text,
  never by database-generated serial IDs.
- Questions are human-authored and distributed across all unique source papers.
  A binary relevance label means that a passage contains direct evidence for
  the question; multiple chunks can be labeled when the answer needs evidence
  from more than one passage.
- Current labels are curated answer-bearing passages, not yet exhaustive
  judgments of every candidate returned by every system. Treat the scores as
  preliminary. A next annotation pass should judge the pooled top results from
  FTS, dense, and hybrid retrieval, including explicit non-relevant judgments.
- The question set is stratified for basic coverage, not sampled from a real
  user-traffic distribution. Results are macro-averaged over questions, so
  every question currently has equal weight.

## Experiment decisions

- Compare raw FTS, raw dense retrieval, and RRF fusion with the same query set,
  corpus, candidate depth, and local PostgreSQL/pgvector instance. Do not mix
  reranking into this comparison; test it as a separate stage later.
- FTS means PostgreSQL `websearch_to_tsquery('english')` ordered by `ts_rank`;
  it is not BM25. Dense retrieval uses `all-MiniLM-L6-v2` and cosine distance.
- Report Recall@5/10/20, MRR, and binary nDCG@5/10/20. Recall shows whether
  marked evidence is found, MRR emphasizes the first relevant result, and nDCG
  rewards placing relevant evidence near the top.
- Keep the pre-deduplication 30-question run as a historical baseline. Do not
  compare its scores directly with later runs as if only the retrieval method
  changed: both the corpus and labels changed.

## Interpretation guardrails

- The current 50-question set is still small. Its results identify issues to
  investigate, but are not a general claim about retrieval quality.
- Incomplete relevance judgments can make Recall and ranking metrics look
  worse than they are. Pooling and human review are needed before selecting a
  production method.
- The reported hybrid result must be judged against dense retrieval, not only
  FTS; fusion is useful only if it provides a repeatable improvement without
  unacceptable latency or complexity.
