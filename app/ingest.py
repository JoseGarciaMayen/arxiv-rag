import os
import re
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

load_dotenv()

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DB_URL = os.getenv("DATABASE_URL")

# Cleaning
_TRAILING_SECTION_RE = re.compile(
    r'\n\s*(?:references|bibliography|acknowledgements?|appendix)\s*\n',
    re.IGNORECASE,
)
_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_HYPHEN_BREAK_RE = re.compile(r'(\w)-\n(\w)')
_EXCESS_NEWLINE_RE = re.compile(r'\n{3,}')

# First-line patterns that signal a noisy/header chunk
_NOISY_FIRST_LINE_RE = re.compile(
    r'^(?:'
    r'published\s+(?:as\s+)?a?\s*(?:conference|journal|workshop)'  # "Published as a conference paper at..."
    r'|under\s+review\s+as'                                        # "Under review as a conference paper..."
    r'|preprint\b'                                                 # "Preprint."
    r'|proceedings\s+of'                                           # "Proceedings of..."
    r'|\[\d[\d,\s]*\]\s+\w'                                        # "[1] Author..." — reference list entry
    r'|\d+$'                                                       # lone page number
    r')',
    re.IGNORECASE,
)

def setup_db(engine):
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                source TEXT,
                content TEXT,
                embedding vector(384)
            )
        """))
        conn.commit()

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]

    # Lines appearing on ≥30% of pages (min 3) are running headers/footers — strip them.
    threshold = max(3, len(pages) * 0.30)
    line_counts: Counter = Counter()
    for page_text in pages:
        for line in set(page_text.splitlines()):
            if line.strip():
                line_counts[line.strip()] += 1
    running_lines = {line for line, count in line_counts.items() if count >= threshold}

    cleaned: list[str] = []
    for page_text in pages:
        lines = [l for l in page_text.splitlines() if l.strip() not in running_lines]
        cleaned.append("\n".join(lines))

    return "\n".join(cleaned)

def clean_text(raw: str) -> str:
    raw = raw.replace('\x00', '')
    match = _TRAILING_SECTION_RE.search(raw)
    if match:
        raw = raw[:match.start()]
    raw = _HYPHEN_BREAK_RE.sub(r'\1\2', raw)
    raw = _URL_RE.sub('', raw)
    raw = _EXCESS_NEWLINE_RE.sub('\n\n', raw)

    return raw.strip()

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
    )
    return splitter.split_text(text)

def is_valid_chunk(chunk: str) -> bool:
    stripped = chunk.strip()

    if len(stripped) < 100:
        return False

    first_line = stripped.split('\n')[0].strip()
    if _NOISY_FIRST_LINE_RE.match(first_line):
        return False

    words = stripped.split()
    if len(words) < 15:
        return False

    # Reject chunks dominated by citation markers like [1], [2, 3], [12]
    citation_hits = len(re.findall(r'\[\d[\d,\s]*\]', stripped))
    if citation_hits >= 5:
        return False

    # Reject chunks where less than half the characters are alphanumeric
    alnum_ratio = sum(c.isalnum() or c.isspace() for c in stripped) / len(stripped)
    if alnum_ratio < 0.5:
        return False

    # Reject chunks with very short average word length
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 3:
        return False

    return True

def embed_and_store(chunks: list[str], source: str, engine):
    if engine is None:
        engine = create_engine(DB_URL)
    valid_chunks = [c for c in chunks if is_valid_chunk(c)]
    embeddings = EMBED_MODEL.encode(valid_chunks, show_progress_bar=True)
    with engine.connect() as conn:
        for chunk, embedding in zip(valid_chunks, embeddings):
            conn.execute(
                text("INSERT INTO chunks (source, content, embedding) VALUES (:source, :content, :embedding)"),
                {"source": source.replace('\x00', ''), "content": chunk.replace('\x00', ''), "embedding": str(embedding.tolist())}
            )
        conn.commit()


def ingest_text(pdf_path: str):
    engine = create_engine(DB_URL)
    setup_db(engine)
    print(f"Extrayendo texto de {pdf_path}...")
    raw_text = extract_text(pdf_path)
    print(f"Texto extraído: {len(raw_text)} caracteres")
    clean = clean_text(raw_text)
    print(f"Texto limpio: {len(clean)} caracteres")
    chunks = chunk_text(clean)
    print(f"Chunks generados: {len(chunks)}")
    embed_and_store(chunks, source=Path(pdf_path).name, engine=engine)
    print("Ingesta completada.")

if __name__ == "__main__":
    import sys
    ingest_text(sys.argv[1])