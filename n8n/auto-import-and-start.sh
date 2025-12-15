#!/bin/sh
set -e

# Optional helper: import all workflows found under /workflows, then start n8n
if [ -d "/workflows" ]; then
  for f in /workflows/*.json; do
    if [ -f "$f" ]; then
      echo "Importing workflow: $f"
      # Import each JSON file; continue even if one fails
      n8n import:workflow --input "$f" --separate || true
    fi
  done
fi

exec n8n start
