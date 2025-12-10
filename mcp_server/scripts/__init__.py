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
# Por defecto, procesaremos estos símbolos:
DEFAULT_SYMBOLS = ["^IBEX", "^GSPC", "^N225"]