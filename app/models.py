from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
