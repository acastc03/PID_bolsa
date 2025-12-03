#!/bin/bash

# Script para ejecutar el MCP server dentro de Docker
# Claude Desktop ejecuta este script, que redirige stdio al contenedor

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ejecutar el servidor MCP dentro de un contenedor Docker
docker run --rm -i \
  --network host \
  -e DB_HOST=localhost \
  -e DB_PORT=15433 \
  -e DB_NAME=indices \
  -e DB_USER=finanzas \
  -e DB_PASS=finanzas_pass \
  -e PYTHONPATH=/app \
  -v "$PROJECT_ROOT:/app" \
  python:3.11-slim \
  bash -c "
    cd /app && \
    pip install -q mcp yfinance feedparser pandas psycopg2-binary python-dotenv scikit-learn xgboost lightgbm catboost prophet && \
    python /app/mcp_server_claude/server.py
  "
