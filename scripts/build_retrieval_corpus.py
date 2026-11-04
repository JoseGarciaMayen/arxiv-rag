"""Build a deterministic, local retrieval corpus from the repository PDFs."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PDF_DIR = ROOT / "docs"
OUTPUT = ROOT / "benchmark" / "corpus.jsonl"
MANIFEST = ROOT / "benchmark" / "corpus-manifest.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    from app.ingest import chunk_text, clean_text, extract_text, is_valid_chunk

    sources = []
    rows = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        raw = extract_text(str(pdf))
        chunks = [chunk for chunk in chunk_text(clean_text(raw)) if is_valid_chunk(chunk)]
        sources.append({"source": pdf.name, "sha256": sha256(pdf.read_bytes())})
        for ordinal, content in enumerate(chunks):
            content_hash = sha256(content.encode("utf-8"))
            rows.append(
                {
                    "chunk_id": f"{pdf.name}:{content_hash}",
                    "source": pdf.name,
                    "ordinal": ordinal,
                    "content_sha256": content_hash,
                    "content": content,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "preprocessing": {
                    "extractor": "PyMuPDF",
                    "chunk_size": 1500,
                    "chunk_overlap": 150,
                    "validity_filter": "app.ingest.is_valid_chunk",
                },
                "sources": sources,
                "chunk_count": len(rows),
                "corpus_sha256": sha256(OUTPUT.read_bytes()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} chunks from {len(sources)} PDFs to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
