# mcp_server/scripts/assets.py
"""Módulo de gestión de símbolos y mercados financieros.

Proporciona utilidades para convertir nombres de mercados legibles
en símbolos válidos de Yahoo Finance, facilitando el uso de la API.
"""

from enum import Enum

# Mapa de alias "humanos" -> símbolo real de yfinance
# Permite usar nombres como "IBEX35" en lugar de "^IBEX"
SYMBOL_ALIASES = {
    # === EUROPA ===
    # España
    "IBEX35": "^IBEX",
    "IBEX": "^IBEX",
    # Reino Unido
    "FTSE100": "^FTSE",
    "FTSE": "^FTSE",
    # Alemania
    "DAX": "^GDAXI",
    "DAX40": "^GDAXI",
    # Francia
    "CAC40": "^FCHI",
    "CAC": "^FCHI",
    # Italia
    "FTSEMIB": "FTSEMIB.MI",
    "ITALY40": "FTSEMIB.MI",
    # Europa general
    "EUROSTOXX50": "^STOXX50E",
    "STOXX50": "^STOXX50E",
    
    # === AMÉRICA ===
    # EE.UU. - Principales
    "SP500": "^GSPC",
    "S&P500": "^GSPC",
    "SPX": "^GSPC",
    "DOW": "^DJI",
    "DOWJONES": "^DJI",
    "DJI": "^DJI",
    "NASDAQ": "^IXIC",
    "NASDAQ100": "^NDX",
    "NQ100": "^NDX",
    # EE.UU. - Otros
    "RUSSELL2000": "^RUT",
    "RUSSELL": "^RUT",
    "VIX": "^VIX",  # Índice de volatilidad
    # Brasil
    "BOVESPA": "^BVSP",
    "IBOVESPA": "^BVSP",
    # México
    "IPC": "^MXX",
    "MEXICO": "^MXX",
    
    # === ASIA-PACÍFICO ===
    # Japón
    "NIKKEI": "^N225",
    "NIKKEI225": "^N225",
    # China
    "SSE": "000001.SS",  # Shanghai Composite
    "SHANGHAI": "000001.SS",
    "HANGSENG": "^HSI",  # Hong Kong
    "HSI": "^HSI",
    # India
    "SENSEX": "^BSESN",
    "BSE": "^BSESN",
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    # Australia
    "ASX200": "^AXJO",
    "ASX": "^AXJO",
    # Corea del Sur
    "KOSPI": "^KS11",
    
    # === SECTORES (EE.UU.) ===
    "TECH": "^IXIC",  # Technology (NASDAQ)
    "FINANCE": "^SP500-40",  # Financials
    "ENERGY": "^GSPE",  # Energy
    "HEALTHCARE": "^SP500-35",  # Healthcare
}


class Market(str, Enum):
    """Enumeración de mercados financieros soportados.
    
    Define los mercados disponibles para consultas en los endpoints de la API.
    Los valores corresponden a las claves del diccionario SYMBOL_ALIASES.
    """
    # Europa
    ibex35 = "IBEX35"          # Índice bursátil español
    ftse100 = "FTSE100"        # FTSE 100 Reino Unido
    dax = "DAX"                # DAX 40 Alemania
    cac40 = "CAC40"            # CAC 40 Francia
    ftsemib = "FTSEMIB"        # FTSE MIB Italia
    eurostoxx50 = "EUROSTOXX50"  # Euro Stoxx 50 Europa
    
    # América
    sp500 = "SP500"            # S&P 500 estadounidense
    dow = "DOW"                # Dow Jones Industrial
    nasdaq = "NASDAQ"          # NASDAQ Composite
    nasdaq100 = "NASDAQ100"    # NASDAQ 100
    russell2000 = "RUSSELL2000"  # Russell 2000 small caps
    vix = "VIX"                # Índice de volatilidad
    bovespa = "BOVESPA"        # Ibovespa Brasil
    ipc = "IPC"                # IPC México
    
    # Asia-Pacífico
    nikkei = "NIKKEI"          # Nikkei 225 Japón
    hangseng = "HANGSENG"      # Hang Seng Hong Kong
    shanghai = "SHANGHAI"      # Shanghai Composite China
    sensex = "SENSEX"          # BSE Sensex India
    nifty50 = "NIFTY50"        # Nifty 50 India
    asx200 = "ASX200"          # ASX 200 Australia
    kospi = "KOSPI"            # KOSPI Corea del Sur


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