#!/bin/bash

set -e

echo "=== ArXiv RAG Setup ==="

if [ ! -f .env ]; then
    cp .env.example .env
    python3 - <<'PY'
from getpass import getpass
from pathlib import Path

env_path = Path(".env")
lines = env_path.read_text().splitlines()
provider = input("Chat provider [groq/deepseek] (groq): ").strip().lower() or "groq"

if provider == "groq":
    values = {
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": getpass("Groq API key: "),
    }
elif provider == "deepseek":
    values = {
        "LLM_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": getpass("DeepSeek API key: "),
        "DEEPSEEK_MODEL": input("Exact DeepSeek API model ID: ").strip(),
    }
else:
    raise SystemExit("Choose either 'groq' or 'deepseek'.")

required_key = "GROQ_API_KEY" if provider == "groq" else "DEEPSEEK_API_KEY"
if not values[required_key]:
    raise SystemExit(f"{required_key} cannot be empty.")
if provider == "deepseek" and not values["DEEPSEEK_MODEL"]:
    raise SystemExit("DEEPSEEK_MODEL cannot be empty; enter the exact model ID from your API account.")

for name, value in values.items():
    prefix = f"{name}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = prefix + value
            break
    else:
        lines.append(prefix + value)

env_path.write_text("\n".join(lines) + "\n")
PY
    echo "✓ .env created"
fi

echo "Building and starting services..."
docker compose up --build -d

echo ""
echo "✓ Setup complete"
echo "  App:  http://localhost:3000"
echo "  Docs: http://localhost:8000/docs"
echo ""
echo "To stop: docker compose down"
echo "To start again: docker compose up"
