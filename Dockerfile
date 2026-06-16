FROM python:3.14-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

COPY app/ app/

CMD ["/bin/sh", "-c", "uv run uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-10000}"]