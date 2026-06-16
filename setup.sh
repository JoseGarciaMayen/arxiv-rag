#!/bin/bash

set -e

echo "=== ArXiv RAG Setup ==="

if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    read -p "Enter your Groq API key (get one free at console.groq.com): " api_key
    sed -i "s/your_groq_api_key_here/$api_key/" .env
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