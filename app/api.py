import os
import uuid, pathlib

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from app.ingest import setup_db, embed_and_store, chunk_text
from app.query import query

load_dotenv()

app = FastAPI(title="ArXiv RAG API")
engine = create_engine(os.getenv("DATABASE_URL"))

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

    safe_name = f"{uuid.uuid4()}.pdf"
    tmp_path = f"/tmp/{safe_name}"
    with open(tmp_path, "wb") as f:
        f.write(content)

    from app.ingest import extract_text, clean_text
    raw_text = extract_text(tmp_path)
    chunks = chunk_text(clean_text(raw_text))
    embed_and_store(chunks, source=file.filename, engine=engine)

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
    return {"message": f"{filename} eliminado"}