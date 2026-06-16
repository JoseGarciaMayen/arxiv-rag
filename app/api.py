import json
import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import create_engine, text

from app.ingest import chunk_text, clean_text, embed_and_store, extract_text, setup_db
from app.query import query, stream_query

load_dotenv()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="ArXiv RAG API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise RuntimeError("DATABASE_URL environment variable is not set")
engine = create_engine(db_url)


class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []


class QueryResponse(BaseModel):
    answer: str


@app.on_event("startup")
async def startup():
    setup_db(engine)


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only accepts pdf")

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw_text = extract_text(tmp_path)
        chunks = chunk_text(clean_text(raw_text))
        stored = embed_and_store(chunks, source=file.filename, engine=engine)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        os.remove(tmp_path)

    return {"message": "Ingest completed", "chunks": stored}


@app.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question can't be empty")

    answer = query(body.question, engine, body.history)
    return QueryResponse(answer=answer)


@app.post("/query/stream")
@limiter.limit("30/minute")
async def query_stream(request: Request, body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question can't be empty")

    def generate():
        for token in stream_query(body.question, engine, body.history):
            yield f"data: {json.dumps({'content': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2"}


@app.get("/documents")
async def list_documents():
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT source, COUNT(*) as chunks FROM chunks GROUP BY source ORDER BY source")
        )
        docs = [{"name": row.source, "chunks": row.chunks} for row in result]
    return {"documents": docs}


@app.delete("/document/{filename}")
async def delete_document(filename: str):
    with engine.connect() as conn:
        result = conn.execute(
            text("DELETE FROM chunks WHERE source = :source"), {"source": filename}
        )
        conn.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return {"message": f"{filename} deleted"}
