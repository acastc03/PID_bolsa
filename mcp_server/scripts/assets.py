# mcp_server/scripts/assets.py
"""Módulo de gestión de símbolos y mercados financieros.

Proporciona utilidades para convertir nombres de mercados legibles
en símbolos válidos de Yahoo Finance, facilitando el uso de la API.
"""

from enum import Enum

# Mapa de alias "humanos" -> símbolo real de yfinance
# Permite usar nombres como "IBEX35" en lugar de "^IBEX"
SYMBOL_ALIASES = {
    # Índice español
    "IBEX35": "^IBEX",
    "IBEX": "^IBEX",
    # S&P 500 (EE.UU.)
    "SP500": "^GSPC",
    "S&P500": "^GSPC",
    "SPX": "^GSPC",
    # NASDAQ Composite y 100 (EE.UU.)
    "NASDAQ": "^IXIC",
    "NASDAQ100": "^NDX",
    "NQ100": "^NDX",
    # Nikkei 225 (Japón)
    "NIKKEI": "^N225",
    "NIKKEI225": "^N225",
}


class Market(str, Enum):
    """Enumeración de mercados financieros soportados.
    
    Define los mercados disponibles para consultas en los endpoints de la API.
    Los valores corresponden a las claves del diccionario SYMBOL_ALIASES.
    """
    ibex35 = "IBEX35"    # Índice bursátil español
    sp500 = "SP500"      # S&P 500 estadounidense
    nasdaq = "NASDAQ"    # NASDAQ Composite
    nikkei = "NIKKEI"    # Nikkei 225 japonés


def resolve_symbol(market_or_symbol: str) -> str:
    """Convierte nombres de mercado legibles en símbolos de Yahoo Finance.
    
    Permite flexibilidad al usuario: puede pasar "IBEX35", "SP500", etc.,
    o directamente un símbolo de yfinance como "^IBEX", "^GSPC".

    Args:
        market_or_symbol: Nombre del mercado (ej: "IBEX35", "SP500") 
                         o símbolo directo (ej: "^IBEX", "^GSPC")

    Returns:
        str: Símbolo válido de Yahoo Finance (ej: "^IBEX", "^GSPC")
        
    Raises:
        ValueError: Si el mercado/símbolo no es reconocido
        
    Examples:
        >>> resolve_symbol("IBEX35")
        '^IBEX'
        >>> resolve_symbol("^GSPC")
        '^GSPC'
    """
    key = market_or_symbol.strip().upper()

    # 1) Si es un alias conocido del diccionario
    if key in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[key]

    # 2) Si ya es un símbolo válido tipo ^IBEX (bypass directo)
    if key.startswith("^"):
        return key

    # 3) No reconocido - lanzar error con opciones disponibles
    raise ValueError(
        f"Índice desconocido: {market_or_symbol}. "
        f"Opciones válidas: {', '.join(SYMBOL_ALIASES.keys())} o símbolos que empiecen por ^"
    )