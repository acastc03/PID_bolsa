"""Paquete de scripts para el servidor MCP Finance.

Proporciona módulos para:
- Descarga de datos (prices, news)
- Cálculo de indicadores técnicos
- Entrenamiento y predicción con modelos ML
- Validación y reporting de resultados

Configuración:
- Logger configurado a nivel INFO por defecto
- Símbolos por defecto: Principales índices globales (Europa, América, Asia)
"""

import logging

# Configurar logger para todos los módulos del paquete
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_finance")

# Símbolos por defecto para operaciones batch
# Incluye principales índices de cada región
DEFAULT_SYMBOLS = [
    "^IBEX",   # España - IBEX 35
    "^GSPC",   # USA - S&P 500
    "^IXIC",   # USA - NASDAQ
    "^N225",   # Japón - Nikkei 225
    "^FTSE",   # UK - FTSE 100
    "^GDAXI",  # Alemania - DAX
]