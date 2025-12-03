#!/bin/bash

# Build and run MCP server using pre-built Docker image
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Build image if it doesn't exist or if --rebuild flag is passed
if [[ "$1" == "--rebuild" ]] || ! docker image inspect mcp-finance-server:latest &> /dev/null; then
    echo "Building MCP server Docker image..." >&2
    docker build -t mcp-finance-server:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"
fi

# Run the container with stdio passthrough
docker run --rm -i \
  --network host \
  -e DB_HOST=localhost \
  -e DB_PORT=15433 \
  -e DB_NAME=indices \
  -e DB_USER=finanzas \
  -e DB_PASS=finanzas_pass \
  -e PYTHONPATH=/app \
  -v "$PROJECT_ROOT/data:/app/data" \
  mcp-finance-server:latest
