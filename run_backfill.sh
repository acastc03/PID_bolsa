#!/bin/bash
# Script para ejecutar backfill_predictions desde DENTRO del contenedor Docker

echo "🚀 Ejecutando backfill de predicciones..."
echo ""
echo "⚠️  ADVERTENCIA: Este proceso tiene look-ahead bias"
echo "   Las predicciones históricas usan información del futuro"
echo ""

docker exec -it mcp_finance python -m scripts.backfill_predictions

echo ""
echo "✅ Backfill completado"
