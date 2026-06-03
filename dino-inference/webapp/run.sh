#!/usr/bin/env bash
# Launch the DINOv3 playground. Run from the dino-inference/ root.
set -e
cd "$(dirname "$0")/.."
exec uv run uvicorn webapp.app:app --reload --port "${PORT:-8000}"
