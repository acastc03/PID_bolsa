from fastapi import FastAPI

from scripts.fetch_data import update_prices_for_symbol
from scripts.news import update_news_for_symbols
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
def update_prices(symbol: str = "^IBEX", period: str = "2y"):
    rows = update_prices_for_symbol(symbol, period)
    return {
        "symbol": symbol,
        "period": period,
        "rows_inserted_or_updated": rows,
    }


@app.get("/update_news")
def update_news(symbols: str = "^IBEX,^GSPC"):
    """
    Descarga noticias recientes para una lista de símbolos separada por comas
    y las guarda en la BD.
    """
    symbols_list = [s.strip() for s in symbols.split(",") if s.strip()]
    inserted = update_news_for_symbols(symbols_list)
    return {
        "symbols": symbols_list,
        "news_inserted": inserted,
    }


@app.get("/compute_indicators")
def compute_indicators(symbol: str = "^IBEX"):
    """
    Calcula indicadores técnicos (SMA20, SMA50, vol_20, RSI14)
    a partir de la tabla 'prices' y los guarda en 'indicators'.
    Pensado para ser llamado desde n8n después de /update_prices.
    """
    rows = compute_indicators_for_symbol(symbol)
    return {
        "symbol": symbol,
        "rows_updated": rows,
    }


@app.get("/compute_signals")
def compute_signals(symbol: str = "^IBEX"):
    """
    Calcula señales de trading basadas en reglas (o modelos ML),
    las guarda en la tabla 'signals' y devuelve la última señal.
    Pensado para ser llamado desde n8n después de /compute_indicators.
    """
    result = compute_signals_for_symbol(symbol)
    return result


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
def daily_summary(symbol: str = "^IBEX"):
    """
    Devuelve un resumen diario con:
      - último precio y variación
      - señales simple y ensemble
      - indicadores técnicos principales
      - noticias recientes
      - texto listo para enviar por email
    Pensado para que n8n lo consuma directamente.
    """
    summary = build_daily_summary(symbol)
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