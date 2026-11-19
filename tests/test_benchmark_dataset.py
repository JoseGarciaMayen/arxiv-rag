import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_SOURCE_PDFS = any((ROOT / "docs").glob("*.pdf"))


def test_frozen_corpus_manifest_and_labels_are_consistent():
    manifest = json.loads((ROOT / "benchmark/corpus-manifest.json").read_text())
    corpus_path = ROOT / "benchmark/corpus.jsonl"
    corpus_bytes = corpus_path.read_bytes()
    corpus = [json.loads(line) for line in corpus_bytes.decode().splitlines()]
    questions = [
        json.loads(line) for line in (ROOT / "benchmark/queries.jsonl").read_text().splitlines()
    ]

    assert hashlib.sha256(corpus_bytes).hexdigest() == manifest["corpus_sha256"]
    assert len(corpus) == manifest["chunk_count"]
    assert len({row["chunk_id"] for row in corpus}) == len(corpus)
    assert len(questions) == 50
    assert len({row["query_id"] for row in questions}) == len(questions)
    corpus_ids = {row["chunk_id"] for row in corpus}
    assert all(
        question["relevant_chunk_ids"] and set(question["relevant_chunk_ids"]) <= corpus_ids
        for question in questions
    )
    assert manifest["excluded_duplicates"] == [
        {"source": "rag.pdf", "duplicate_of": "05_rag_original_2020.pdf"}
    ]
    assert len({source["sha256"] for source in manifest["sources"]}) == len(manifest["sources"])


@pytest.mark.skipif(
    not HAS_SOURCE_PDFS,
    reason="source PDFs are not tracked, so the corpus cannot be regenerated here",
)
def test_corpus_builder_is_deterministic():
    corpus_before = (ROOT / "benchmark/corpus.jsonl").read_bytes()
    manifest_before = (ROOT / "benchmark/corpus-manifest.json").read_bytes()
    subprocess.run([sys.executable, "scripts/build_retrieval_corpus.py"], cwd=ROOT, check=True)
    assert (ROOT / "benchmark/corpus.jsonl").read_bytes() == corpus_before
    assert (ROOT / "benchmark/corpus-manifest.json").read_bytes() == manifest_before
