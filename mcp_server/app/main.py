from fastapi import FastAPI, HTTPException, Query
from datetime import datetime
from psycopg2 import Error as PsycopgError

from scripts.assets import Market, resolve_symbol

from scripts.save_predictions import save_daily_predictions

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

from scripts.validate_predictions import (
    validate_predictions_for_date,
    validate_predictions_yesterday,
)

from scripts.reporting import build_daily_summary

from scripts.model_storage import delete_old_models, get_model_info

from scripts.model_evaluation import (
    get_model_performance_report,
    should_retrain_models,
)

app = FastAPI(
    title="MCP Finance Server",
    version="0.1.0",
    description="API para datos de mercado, noticias y modelos de predicción.",
)


from datetime import date  

# ===================================================================
# 1. ENDPOINTS DE UTILIDAD Y SALUD
# ===================================================================

@app.get("/health")
def health():
    """Chequeo rápido de que la API está viva."""
    return {"status": "ok"}


@app.get("/markets")
def list_markets():
    """
    Lista todos los mercados financieros soportados.
    
    Devuelve información sobre los 3 índices principales:
    IBEX35 (España), SP500 (USA), NIKKEI (Japón).
    """
    from scripts.assets import SYMBOL_ALIASES, Market
    
    # Organizar mercados por región
    markets_by_region = {
        "europe": [
            {"name": "IBEX35", "description": "IBEX 35 - España", "symbol": "^IBEX"},
        ],
        "americas": [
            {"name": "SP500", "description": "S&P 500 - USA", "symbol": "^GSPC"},
        ],
        "asia_pacific": [
            {"name": "NIKKEI", "description": "Nikkei 225 - Japón", "symbol": "^N225"},
        ],
    }
    
    return {
        "total_markets": sum(len(markets) for markets in markets_by_region.values()),
        "markets_by_region": markets_by_region,
        "available_in_enum": [m.value for m in Market],
    }


# ===================================================================
# 2. ENDPOINTS DE INGESTA DE DATOS (ETL - Extract)
# ===================================================================

@app.get("/update_prices")
def update_prices(market: Market = Market.ibex35, period: str = "1mo"):
    """
    Actualiza precios históricos para el índice seleccionado.
    
    Mercados soportados:
    - Europa: IBEX35 (España)
    - América: SP500 (USA)
    - Asia-Pacífico: NIKKEI (Japón)
    
    Period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
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
    markets: str = "IBEX35",
    when: str = "7d",
    days: int = 7,
    limit_rss: int = 10,
    limit_yf: int = 10,
):
    """
    Descarga noticias para una lista de índices separados por comas.
    
    Ejemplos de mercados:
    - Un mercado: IBEX35
    - Varios: IBEX35,SP500,NIKKEI
    
    Las noticias se guardan en la tabla 'news'.
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
    """
    Calcula indicadores técnicos (SMA, RSI, volatilidad) para un mercado.
    
    Mercados: IBEX35, SP500, NIKKEI. Los indicadores se guardan en la tabla 'indicators'.
    """
    try:
        symbol = resolve_symbol(market.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows = compute_indicators_for_symbol(symbol)
    return {"market": market.value, "symbol": symbol, "rows_updated": rows}

@app.get("/compute_signals")
def compute_signals(market: Market = Market.ibex35):
    """
    Genera señales de trading simples basadas en indicadores técnicos.
    
    Señales: +1 (COMPRA), 0 (NEUTRAL), -1 (VENTA)
    Mercados: IBEX35, SP500, NIKKEI.
    """
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

    Además, guarda las predicciones de cada modelo en la tabla ml_predictions
    para llevar un histórico diario.
    """
    # 1) Obtener resultados del ensemble (tal como ya hacías)
    result = predict_ensemble(symbol)

    # 2) Construir el diccionario de predicciones por modelo.
    #    Aquí asumimos que `result["ml_models"]` es algo tipo:
    #    [
    #      {
    #        "model_name": "LinearRegression",
    #        "prediction_next_day": 15920.52,
    #        "signal_next_day": -1,
    #        ...
    #      },
    #      ...
    #    ]
    #    y que quieres guardar:
    #    - prediction_next_day como predicted_value (precio)
    #    - signal_next_day como predicted_signal (+1, 0, -1)
    predictions_dict = {}
    for m in result.get("ml_models", []):
        model_name = m.get("model_name") or m.get("name")
        price = m.get("prediction_next_day")
        signal = m.get("signal_next_day")
        if model_name is not None:
            predictions_dict[model_name] = {
                "price": price,
                "signal": signal,
            }

    # 3) También guardamos la señal del ensemble como un "modelo" más
    #    para poder evaluarlo luego frente al resto.
    #    Opcionalmente, podemos guardar como precio del ensemble la media
    #    de los prediction_next_day de los modelos individuales.
    if "signal_ensemble" in result:
        prices = [
            m.get("prediction_next_day")
            for m in result.get("ml_models", [])
            if m.get("prediction_next_day") is not None
        ]
        avg_price = sum(prices) / len(prices) if prices else None

        predictions_dict["ensemble"] = {
            "price": avg_price,                  # puede ser None si no quieres guardar precio
            "signal": result["signal_ensemble"], # señal agregada del ensemble
        }

    # 4) Definimos fechas: run_date = hoy; prediction_date = hoy+1 (o hoy, según tu lógica).
    today = date.today()
    prediction_date = today  # o calcula mañana si tu modelo siempre predice la sesión siguiente
    run_date = today

    # 5) Guardar en la BD (solo si hay algo que guardar)
    if predictions_dict:
        save_daily_predictions(
            symbol=symbol,
            prediction_date=prediction_date,
            run_date=run_date,
            predictions=predictions_dict,
        )

    # 6) Devolver la respuesta original
    return {
        "symbol": symbol,
        **result,
    }

# ===================================================================
# 4.b ENDPOINT DE VALIDACIÓN DE PREDICCIONES
# ===================================================================

@app.post("/validate_predictions")
def validate_predictions(
    date_str: str | None = Query(
        default=None,
        description="Fecha a validar en formato YYYY-MM-DD; si se omite, se usa ayer",
    )
):
    """
    Valida las predicciones guardadas en ml_predictions.

    - Si se pasa date_str (YYYY-MM-DD), usa esa fecha como prediction_date.
    - Si no se pasa, valida las predicciones de 'ayer'.
    """
    # 1) Validar formato de fecha (ya lo hacías bien)
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Formato de fecha inválido, usa YYYY-MM-DD",
            )
        try:
            result = validate_predictions_for_date(target_date)
        except PsycopgError:
            # Por si en el futuro cambias validate_predictions_for_date
            raise HTTPException(
                status_code=500,
                detail="Error de base de datos al validar predicciones",
            )
    else:
        try:
            result = validate_predictions_yesterday()
        except PsycopgError:
            raise HTTPException(
                status_code=500,
                detail="Error de base de datos al validar predicciones",
            )

    # 2) Si la función devolvió un error de BD en forma de dict → 500 controlado
    if result.get("error") == "database_error":
        raise HTTPException(
            status_code=500,
            detail=result.get("message", "Error de base de datos al validar predicciones"),
        )

    # 3) Si no hay precios para esa fecha → 404 controlado (fecha no disponible)
    if not result.get("symbols_with_price"):
        raise HTTPException(
            status_code=404,
            detail=result.get("message", "No hay precios en 'prices' para esa fecha"),
        )

    # 4) Caso OK
    return result

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


@app.post("/tune_models")
def tune_models_endpoint(symbol: str = "^IBEX", n_iter: int = 20):
    """
    🔍 Optimiza hiperparámetros de todos los modelos ML usando Bayesian Optimization.
    
    Este endpoint:
    1. Ejecuta hyperparameter tuning para cada modelo (RandomForest, XGBoost, SVR, LightGBM, CatBoost)
    2. Encuentra los mejores parámetros usando validación cruzada
    3. Entrena nuevos modelos con parámetros optimizados
    4. Guarda los modelos y parámetros en disco
    5. Retorna las métricas de mejora
    
    Args:
        symbol: Símbolo del activo (^IBEX, ^GSPC, ^N225)
        n_iter: Número de iteraciones de Bayesian optimization (default: 20, recomendado: 30-50 para mejores resultados)
    
    Returns:
        Diccionario con modelos optimizados y métricas de mejora
        
    ⚠️ ADVERTENCIA: Este proceso puede tardar varios minutos (5-15 min dependiendo de n_iter)
    """
    import time
    start_time = time.time()
    
    # Entrenar con hyperparameter tuning activado
    result = predict_ensemble(symbol, force_retrain=True, tune_hyperparams=True)
    
    elapsed_time = time.time() - start_time
    
    # Contar cuántos modelos fueron optimizados
    tuned_models = [m for m in result["ml_models"] if m.get("from_cache") == False]
    
    return {
        "message": f"✅ Hyperparameter tuning completado para {symbol}",
        "symbol": symbol,
        "models_tuned": len(tuned_models),
        "elapsed_time_seconds": round(elapsed_time, 2),
        "elapsed_time_minutes": round(elapsed_time / 60, 2),
        "signal_ensemble": result["signal_ensemble"],
        "ml_models": result["ml_models"],
        "info": f"Modelos optimizados con {n_iter} iteraciones de Bayesian optimization"
    }


# ===================================================================
# 6. ENDPOINTS DE REPORTING (Salida)
# ===================================================================

@app.get("/daily_summary")
def daily_summary(market: Market = Market.ibex35, include_ml: bool = True):
    """
    Genera un resumen completo del día para un mercado.
    
    Incluye:
    - Precio actual y variación
    - Indicadores técnicos (SMA, RSI, volatilidad)
    - Señales de trading
    - Últimas noticias
    - Rendimiento de modelos ML (opcional)
    
    Mercados: IBEX35, SP500, NIKKEI.
    Ideal para reportes diarios automatizados.
    """
    try:
        symbol = resolve_symbol(market.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    summary = build_daily_summary(symbol, include_ml_performance=include_ml)
    # añadimos info del market original
    summary["market"] = market.value
    return summary


@app.get("/model_performance")
def model_performance(
    symbol: str = "^IBEX",
    days: int = 30
):
    """
    Genera un reporte completo de rendimiento de modelos ML.
    
    Analiza predicciones validadas de los últimos N días para:
    - Ver MAE y RMSE de cada modelo
    - Identificar modelos con bajo rendimiento
    - Determinar qué modelos necesitan reentrenamiento
    
    Útil para monitorizar la salud de los modelos antes de reentrenar.
    """
    from datetime import date, timedelta
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    report = get_model_performance_report(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    return report


@app.get("/should_retrain")
def should_retrain(
    symbol: str = "^IBEX",
    mae_threshold: float = 200.0
):
    """
    Determina si los modelos necesitan reentrenamiento.
    
    Analiza los últimos 7 días de predicciones validadas y recomienda
    si es momento de reentrenar basándose en:
    - MAE promedio por encima del umbral
    - Número de modelos con bajo rendimiento
    - Inconsistencia en predicciones
    
    Respuesta incluye:
    - should_retrain: bool
    - reasons: lista de razones
    - models_to_retrain: modelos específicos que necesitan mejora
    - detailed_report: análisis completo
    """
    analysis = should_retrain_models(symbol, mae_threshold)
    return analysis


@app.post("/validate_and_retrain")
def validate_and_retrain(
    date_str: str | None = Query(
        default=None,
        description="Fecha a validar en formato YYYY-MM-DD; si se omite, se usa ayer",
    ),
    symbol: str = "^IBEX"
):
    """
    FLUJO DIARIO COMPLETO: Valida predicciones de ayer y reentrena modelos.
    
    Workflow:
    1. Valida las predicciones del día especificado (o ayer por defecto)
    2. Reentrena todos los modelos ML con los datos actualizados
    3. Limpia modelos antiguos (mantiene últimos 7 días)
    
    Este endpoint está diseñado para ejecutarse diariamente desde n8n.
    
    Returns:
        - validation_result: resultado de la validación
        - retrain_result: resultado del reentrenamiento
        - summary: resumen del proceso
    """
    # 1) VALIDAR PREDICCIONES
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Formato de fecha inválido, usa YYYY-MM-DD",
            )
        validation_result = validate_predictions_for_date(target_date)
    else:
        validation_result = validate_predictions_yesterday()
    
    # Verificar si la validación fue exitosa
    if validation_result.get("error"):
        raise HTTPException(
            status_code=500,
            detail=f"Error en validación: {validation_result.get('message')}"
        )
    
    if not validation_result.get("symbols_with_price"):
        raise HTTPException(
            status_code=404,
            detail="No hay precios para validar en esa fecha"
        )
    
    # 2) REENTRENAR MODELOS
    try:
        retrain_result = predict_ensemble(symbol, force_retrain=True)
        
        # 3) LIMPIAR MODELOS ANTIGUOS
        deleted = delete_old_models(symbol, keep_latest=7)
        
        return {
            "validation": {
                "target_date": validation_result["target_date"],
                "symbols_validated": validation_result["symbols_with_price"],
                "predictions_updated": validation_result["rows_updated"],
            },
            "retrain": {
                "models_retrained": len(retrain_result["ml_models"]),
                "signal_ensemble": retrain_result["signal_ensemble"],
                "models": retrain_result["ml_models"],
            },
            "cleanup": {
                "old_models_deleted": deleted,
            },
            "summary": {
                "status": "success",
                "message": f"✅ Validadas {validation_result['rows_updated']} predicciones y reentrenados {len(retrain_result['ml_models'])} modelos",
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el reentrenamiento: {str(e)}"
        )

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
