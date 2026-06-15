import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.query import search_chunks, query

load_dotenv()


def build_eval_dataset(questions: list[str], engine) -> Dataset:
    rows = []
    for question in questions:
        chunks = search_chunks(question, engine)
        answer = query(question, engine)
        rows.append({
            "question": question,
            "answer": answer,
            "contexts": [c["content"] for c in chunks],
        })
    return Dataset.from_list(rows)


def run_evaluation(questions: list[str]):
    engine = create_engine(os.getenv("DATABASE_URL"))

    llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=120,
    ))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    print("Generating answers...")
    dataset = build_eval_dataset(questions, engine)

    print("Evaluating with RAGAS...")

    result = evaluate(
        dataset,
        metrics=[answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n=== RAGAS Results ===")
    print(result)
    return result


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "What is YOLO and how does it work?",
        "How does YOLO compare to R-CNN in speed and accuracy?",
        "What are the main limitations of YOLO?",
    ]
    run_evaluation(questions)