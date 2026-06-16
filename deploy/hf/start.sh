#!/bin/sh
set -e

# Internal API process (nginx proxies /api/ here)
.venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 8000 &

# Public process on the HF Spaces port
exec nginx -g 'daemon off;'
