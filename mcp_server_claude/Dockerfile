FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY mcp_server_claude/requirements.txt /tmp/mcp_requirements.txt
COPY mcp_server/requirements.txt /tmp/server_requirements.txt

RUN pip install --no-cache-dir \
    "mcp[cli]>=1.0.0" \
    yfinance \
    feedparser \
    pandas \
    psycopg2-binary \
    python-dotenv \
    scikit-learn \
    xgboost \
    lightgbm \
    catboost \
    prophet

# Copy application code
COPY mcp_server/ /app/mcp_server/
COPY mcp_server_claude/ /app/mcp_server_claude/

# Set Python path
ENV PYTHONPATH=/app

# Run MCP server
CMD ["python", "/app/mcp_server_claude/server.py"]
