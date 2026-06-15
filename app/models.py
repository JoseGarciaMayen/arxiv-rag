from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
