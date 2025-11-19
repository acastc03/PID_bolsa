from fastapi import FastAPI, HTTPException

from scripts.assets import Market, resolve_symbol

from scripts.fetch_data import update_prices_for_symbol

from scripts.news import (
    fetch_and_store_news_rss,
    fetch_and_store_news_yf,
    update_news_for_symbols,
)
from scripts.indicators import compute_indicators_for_symbol

from scripts.models import (
    compute_signals_for_symbol,
    predict_simple,
    predict_ensemble,
)
from scripts.reporting import build_daily_summary

app = FastAPI(
    title="MCP Finance Server",
    version="0.1.0",
    description="API para datos de mercado, noticias y modelos de predicción.",
)


# -------------------------------------------------------------------
# Endpoints básicos / de servicio (para n8n y pruebas)
# -------------------------------------------------------------------


@app.get("/health")
def health():
    """Chequeo rápido de que la API está viva."""
    return {"status": "ok"}


@app.get("/update_prices")
def update_prices(market: Market = Market.ibex35, period: str = "1mo"):
    """
    Actualiza precios históricos para el índice seleccionado (IBEX35, SP500, NASDAQ, NIKKEI).
    """
    try:
        symbol = resolve_symbol(market.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows = update_prices_for_symbol(symbol, period)
    return {
        "market": market.value,
        "symbol": symbol,
        "period": period,
        "rows_inserted_or_updated": rows,
    }


@app.get("/update_news")
def update_news(
    markets: str = "IBEX35,SP500",
    when: str = "7d",
    days: int = 7,
    limit_rss: int = 10,
    limit_yf: int = 10,
):
    """
    Descarga noticias para una lista de índices separados por comas
    (ej: IBEX35,SP500,NASDAQ) y las guarda en la tabla 'news'.
    """
    market_list = [m.strip() for m in markets.split(",") if m.strip()]

    # convertimos cada market a símbolo real
    symbols = []
    for m in market_list:
        try:
            symbols.append(resolve_symbol(m))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    result = update_news_for_symbols(
        symbols,
        when=when,
        days_back=days,
        max_items_rss=limit_rss,
        max_items_yf=limit_yf,
    )

    return {
        "markets": market_list,
        "symbols": symbols,
        "total_news_inserted": result["total"],
        "details": result["per_symbol"],
    }

@app.get("/compute_indicators")
def compute_indicators(market: Market = Market.ibex35):
    try:
        symbol = resolve_symbol(market.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows = compute_indicators_for_symbol(symbol)
    return {"market": market.value, "symbol": symbol, "rows_updated": rows}


@app.get("/compute_signals")
def compute_signals(market: Market = Market.ibex35):
    try:
        symbol = resolve_symbol(market.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = compute_signals_for_symbol(symbol)
    return {"market": market.value, "symbol": symbol, **result}


@app.get("/predecir_simple")
def predecir_simple(symbol: str = "^IBEX"):
    """
    Devuelve solo la señal 'simple' (+1, 0, -1) para la última fecha disponible.
    """
    sig = predict_simple(symbol)
    return {
        "symbol": symbol,
        "signal_simple": sig,
    }


@app.get("/predecir_ensemble")
def predecir_ensemble(symbol: str = "^IBEX"):
    """
    Devuelve las señales individuales de cada regla/modelo y la señal final
    por votación (ensemble).
    """
    result = predict_ensemble(symbol)
    return {
        "symbol": symbol,
        **result,
    }


@app.get("/daily_summary")
def daily_summary(market: Market = Market.ibex35):
    try:
        symbol = resolve_symbol(market.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    summary = build_daily_summary(symbol)
    # añadimos info del market original
    summary["market"] = market.value
    return summary


# -------------------------------------------------------------------
# Endpoints “legados” / interactivos (para ChatGPT, pruebas, etc.)
# -------------------------------------------------------------------


@app.get("/indicadores")
def indicadores(symbol: str = "^IBEX"):
    """
    Atajo para recuperar indicadores del símbolo.
    Aquí puedes:
      - o bien tirar directamente de BD
      - o reutilizar la lógica de compute_indicators_for_symbol y luego
        leer de la tabla.
    De momento delegamos en compute_indicators_for_symbol.
    """
    rows = compute_indicators_for_symbol(symbol)
    return {
        "symbol": symbol,
        "rows_updated": rows,
    }


@app.get("/predecir_simple")
def predecir_simple(symbol: str = "^IBEX"):
    """
    Devuelve solo la señal del modelo 'simple' (por ejemplo, LR o CatBoost).
    """
    signal = predict_simple(symbol)
    return {
        "symbol": symbol,
        "signal_simple": signal,
    }


@app.get("/predecir_ensemble")
def predecir_ensemble(symbol: str = "^IBEX"):
    """
    Devuelve las señales de cada modelo del ensemble y la votación final.
    """
    result = predict_ensemble(symbol)
    # result puede ser ya un dict con más info (signals, signal_ensemble, etc.)
    return {
        "symbol": symbol,
        **result,
    }