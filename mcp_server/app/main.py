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
from scripts.model_storage import delete_old_models, get_model_info

app = FastAPI(
    title="MCP Finance Server",
    version="0.1.0",
    description="API para datos de mercado, noticias y modelos de predicción.",
)


# ===================================================================
# 1. ENDPOINTS DE UTILIDAD Y SALUD
# ===================================================================

@app.get("/health")
def health():
    """Chequeo rápido de que la API está viva."""
    return {"status": "ok"}


# ===================================================================
# 2. ENDPOINTS DE INGESTA DE DATOS (ETL - Extract)
# ===================================================================

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


# ===================================================================
# 3. ENDPOINTS DE PROCESAMIENTO (ETL - Transform)
# ===================================================================

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


# ===================================================================
# 4. ENDPOINTS DE MODELOS ML (Predicción)
# ===================================================================

@app.get("/predecir_simple")
def predecir_simple(symbol: str = "^IBEX"):
    """
    Devuelve solo la señal 'simple' (+1, 0, -1) basada en reglas
    para la última fecha disponible.
    """
    sig = predict_simple(symbol)
    return {
        "symbol": symbol,
        "signal_simple": sig,
    }


@app.get("/predecir_ensemble")
def predecir_ensemble_endpoint(symbol: str = "^IBEX"):
    """
    Devuelve las señales individuales de cada modelo ML y la señal final
    por votación (ensemble).
    
    Usa modelos guardados en caché si existen (rápido).
    """
    result = predict_ensemble(symbol)
    return {
        "symbol": symbol,
        **result,
    }


# ===================================================================
# 5. ENDPOINTS DE GESTIÓN DE MODELOS ML
# ===================================================================

@app.get("/model_info")
def model_info(symbol: str = "^IBEX"):
    """
    Obtiene información sobre los modelos guardados.
    Muestra qué modelos existen, sus fechas de entrenamiento y métricas.
    """
    return get_model_info(symbol)


@app.get("/retrain_models")
def retrain_models(symbol: str = "^IBEX"):
    """
    Fuerza el reentrenamiento de todos los modelos ML.
    - Entrena nuevos modelos con los datos más recientes
    - Guarda los nuevos modelos
    - Elimina modelos antiguos (mantiene últimos 7 días)
    
    ORDEN: Llamar después de /compute_indicators
    Útil para ejecutar diariamente desde n8n.
    """
    # Forzar reentrenamiento
    result = predict_ensemble(symbol, force_retrain=True)
    
    # Limpiar modelos antiguos (mantener últimos 7 días)
    deleted = delete_old_models(symbol, keep_latest=7)
    
    return {
        "symbol": symbol,
        "models_retrained": len(result["ml_models"]),
        "old_models_deleted": deleted,
        "signal_ensemble": result["signal_ensemble"],
        "ml_models": result["ml_models"]
    }


@app.get("/predecir_ensemble_force")
def predecir_ensemble_force(symbol: str = "^IBEX"):
    """
    Alias de predecir_ensemble con force_retrain=True.
    Fuerza reentrenamiento de modelos y hace predicción.
    """
    return predict_ensemble(symbol, force_retrain=True)


# ===================================================================
# 6. ENDPOINTS DE REPORTING (Salida)
# ===================================================================

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


# ===================================================================
# 7. ENDPOINTS LEGACY / DEPRECADOS (Mantener por compatibilidad)
# ===================================================================

@app.get("/indicadores")
def indicadores(symbol: str = "^IBEX"):
    """
    [LEGACY] Atajo para recuperar indicadores del símbolo.
    
    NOTA: Es redundante con /compute_indicators.
    Se mantiene por compatibilidad con versiones anteriores.
    """
    rows = compute_indicators_for_symbol(symbol)
    return {
        "symbol": symbol,
        "rows_updated": rows,
    }
