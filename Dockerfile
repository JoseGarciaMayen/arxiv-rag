# Combined image for Hugging Face Spaces (Docker SDK).
# Builds the React frontend, serves it with nginx, and proxies /api/ to an
# internal uvicorn process. Mirrors the docker-compose setup in a single container.

# Stage 1: build the frontend
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: python API + nginx
FROM python:3.14-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Cache models in a deterministic, world-readable location
ENV HF_HOME=/app/.cache/huggingface

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

COPY app/ app/

RUN uv run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
    && chmod -R a+rX /app/.cache

# Static frontend + nginx/start config
COPY --from=frontend /fe/dist /usr/share/nginx/html
COPY deploy/hf/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/hf/start.sh /start.sh
RUN rm -f /etc/nginx/sites-enabled/default && chmod +x /start.sh

EXPOSE 7860

CMD ["/start.sh"]
