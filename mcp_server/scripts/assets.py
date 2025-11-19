# mcp_server/scripts/assets.py

from enum import Enum

# Mapa de alias "humanos" -> símbolo real de yfinance
SYMBOL_ALIASES = {
    "IBEX35": "^IBEX",
    "IBEX": "^IBEX",
    "SP500": "^GSPC",
    "S&P500": "^GSPC",
    "SPX": "^GSPC",
    "NASDAQ": "^IXIC",
    "NASDAQ100": "^NDX",
    "NQ100": "^NDX",
    "NIKKEI": "^N225",
    "NIKKEI225": "^N225",
}


class Market(str, Enum):
    ibex35 = "IBEX35"
    sp500 = "SP500"
    nasdaq = "NASDAQ"
    nikkei = "NIKKEI"


def resolve_symbol(market_or_symbol: str) -> str:
    """
    Convierte lo que el usuario escriba (IBEX35, SP500, NASDAQ, NIKKEI o símbolo directo)
    en el símbolo que entiende yfinance.

    - Si es uno de los alias, devuelve el símbolo mapeado.
    - Si empieza por ^, asumimos que ya es símbolo yfinance (^IBEX, ^GSPC...).
    - Si no lo conocemos, levantamos error.
    """
    key = market_or_symbol.strip().upper()

    # 1) Si es un alias conocido
    if key in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[key]

    # 2) Si ya es un símbolo tipo ^IBEX
    if key.startswith("^"):
        return key

    # 3) Desconocido
    raise ValueError(
        f"Índice desconocido: {market_or_symbol}. "
        f"Opciones válidas: {', '.join(SYMBOL_ALIASES.keys())} o símbolos que empiecen por ^"
    )