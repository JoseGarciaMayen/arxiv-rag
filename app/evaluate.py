import math
import os
import time

from datasets import Dataset
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy
from ragas.run_config import RunConfig
from sqlalchemy import create_engine

from app.query import query, search_chunks

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SLEEP_BETWEEN_QUESTIONS = 65


def evaluate_question(question: str, engine, llm, embeddings) -> float:
    chunks = search_chunks(question, engine)
    answer = query(question, engine)
    dataset = Dataset.from_list(
        [
            {
                "question": question,
                "answer": answer,
                "contexts": [c["content"] for c in chunks],
            }
        ]
    )
    result = evaluate(
        dataset,
        metrics=[answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=120, max_retries=2, max_wait=60),
    )
    score = result["answer_relevancy"]
    return float(score) if not hasattr(score, "__iter__") else float(score[0])


def run_evaluation(questions: list[str]):
    engine = create_engine(os.getenv("DATABASE_URL"))

    llm = LangchainLLMWrapper(
        ChatGroq(
            model=GROQ_MODEL,
            api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=120,
            max_tokens=2048,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    scores = []
    for i, question in enumerate(questions):
        if i > 0:
            print(f"  Waiting {SLEEP_BETWEEN_QUESTIONS}s for TPM window to reset...")
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

        print(f"[{i + 1}/{len(questions)}] {question}")
        score = evaluate_question(question, engine, llm, embeddings)
        if isinstance(score, list):
            scores.extend(score)
        else:
            scores.append(score)
        print(f"  answer_relevancy: {score}")

    # Filter out NaN values before calculating average
    valid_scores = [s for s in scores if not (isinstance(s, float) and math.isnan(s))]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    print("\n=== RAGAS Results ===")
    print(f"answer_relevancy: {avg:.4f}  (avg over {len(questions)} questions)")
    return avg


if __name__ == "__main__":
    import sys

    questions = sys.argv[1:] or [
        "What is BERT?",
        "How does the transformer model work?",
        "What are the main limitations of YOLO?",
    ]
    run_evaluation(questions)
