import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from app.ingest import setup_db, embed_and_store, chunk_text, extract_text, clean_text
from app.query import query

load_dotenv()

app = FastAPI(title="ArXiv RAG API")
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
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only accepts pdf")

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw_text = extract_text(tmp_path)
        chunks = chunk_text(clean_text(raw_text))
        embed_and_store(chunks, source=file.filename, engine=engine)
    finally:
        os.remove(tmp_path)

    return {"message": "Ingest completed", "chunks": len(chunks)}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question can't be empty")

    answer = query(request.question, engine, request.history)
    return QueryResponse(answer=answer)

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
        conn.execute(
            text("DELETE FROM chunks WHERE source = :source"),
            {"source": filename}
        )
        conn.commit()
    return {"message": f"{filename} deleted"}
