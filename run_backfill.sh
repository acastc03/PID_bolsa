#!/bin/bash
# Script para ejecutar backfill_predictions desde DENTRO del contenedor Docker

echo "🚀 Ejecutando backfill de predicciones..."
echo ""
echo "✅ Sistema actualizado: SIN look-ahead bias"
echo "   - Usa as_of_date para filtrar datos por fecha"
echo "   - Reentrena modelos para cada fecha (puede ser lento)"
echo "   - Predicciones válidas para análisis histórico"
echo ""
echo "Edita mcp_server/scripts/backfill_predictions.py para cambiar:"
echo "  - Símbolo (^IBEX, ^GSPC, ^N225)"
echo "  - Rango de fechas (start_date, end_date)"
echo ""

docker exec -it mcp_finance python -m scripts.backfill_predictions

echo ""
echo "✅ Backfill completado"
echo "   Usa /validate_predictions y /model_performance para verificar"
