#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> Checking Ollama..."
if ! curl -sf http://localhost:11434/api/tags > /dev/null; then
  echo "ERROR: Ollama is not running. Start it with: ollama serve"
  exit 1
fi

echo "==> Creating data directories..."
mkdir -p data/logs

echo "==> Starting FastAPI backend on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

sleep 2

echo "==> Starting Streamlit dashboard on port 8501..."
streamlit run dashboard/app.py &
DASH_PID=$!

echo ""
echo "  API:       http://localhost:8000/docs"
echo "  Dashboard: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop both services."

wait $API_PID $DASH_PID
