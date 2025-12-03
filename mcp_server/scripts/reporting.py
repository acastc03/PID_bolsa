# mcp_server/scripts/reporting.py

from datetime import datetime
from typing import Dict, Any, List
from psycopg2 import Error as PsycopgError

from .config import get_db_conn
from . import logger


def _get_latest_price(symbol: str):
    """
    Obtiene el último precio de cierre y el anterior para un símbolo dado,
    junto con la variación absoluta y porcentual.
    """
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Último precio
            cur.execute(
                """
                SELECT date, close
                FROM prices
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1;
                """,
                (symbol,),
            )
            row_last = cur.fetchone()
            if not row_last:
                return None, None, None, None, None

            last_date = row_last["date"]
            last_close = float(row_last["close"])

            # Precio anterior (para calcular variación)
            cur.execute(
                """
                SELECT date, close
                FROM prices
                WHERE symbol = %s
                  AND date < %s
                ORDER BY date DESC
                LIMIT 1;
                """,
                (symbol, last_date),
            )
            row_prev = cur.fetchone()

        if not row_prev:
            prev_close = None
            abs_change = None
            pct_change = None
        else:
            prev_close = float(row_prev["close"])
            abs_change = last_close - prev_close
            pct_change = (abs_change / prev_close) * 100 if prev_close != 0 else None

        return last_date, last_close, prev_close, abs_change, pct_change

    except PsycopgError as e:
        logger.error(f"Error al obtener último precio de {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise
    finally:
        if conn is not None and not conn.closed:
            conn.close()


def _get_indicators_for_date(symbol: str, date):
    """
    Obtiene SMA20, SMA50, vol_20 y RSI14 para un símbolo y fecha concretos.
    """
    if date is None:
        return {"sma_20": None, "sma_50": None, "vol_20": None, "rsi_14": None}

    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sma_20, sma_50, vol_20, rsi_14
                FROM indicators
                WHERE symbol = %s
                  AND date = %s
                LIMIT 1;
                """,
                (symbol, date),
            )
            row = cur.fetchone()

        if not row:
            return {"sma_20": None, "sma_50": None, "vol_20": None, "rsi_14": None}

        return {
            "sma_20": float(row["sma_20"]) if row["sma_20"] is not None else None,
            "sma_50": float(row["sma_50"]) if row["sma_50"] is not None else None,
            "vol_20": float(row["vol_20"]) if row["vol_20"] is not None else None,
            "rsi_14": float(row["rsi_14"]) if row["rsi_14"] is not None else None,
        }

    except PsycopgError as e:
        logger.error(f"Error al obtener indicadores de {symbol} en {date}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise
    finally:
        if conn is not None and not conn.closed:
            conn.close()


def _get_latest_signals(symbol: str):
    """
    Obtiene la última señal simple y ensemble para un símbolo.
    """
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, signal_simple, signal_ensemble
                FROM signals
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1;
                """,
                (symbol,),
            )
            row = cur.fetchone()

        if not row:
            return None, {"simple": None, "ensemble": None}

        return row["date"], {
            "simple": int(row["signal_simple"]) if row["signal_simple"] is not None else None,
            "ensemble": int(row["signal_ensemble"]) if row["signal_ensemble"] is not None else None,
        }

    except PsycopgError as e:
        logger.error(f"Error al obtener señales de {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise
    finally:
        if conn is not None and not conn.closed:
            conn.close()


def _get_recent_news(symbol: str, limit: int = 5) -> list:
    """
    Últimas noticias almacenadas en la tabla 'news' para el símbolo.
    """
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT published_at, title, source, url
                FROM news
                WHERE symbol = %s
                ORDER BY published_at DESC
                LIMIT %s;
                """,
                (symbol, limit),
            )
            rows = cur.fetchall()

        news_list: List[Dict[str, Any]] = []
        for r in rows:
            news_list.append(
                {
                    "published_at": (
                        r["published_at"].isoformat()
                        if isinstance(r["published_at"], datetime)
                        else str(r["published_at"])
                    ),
                    "title": r["title"],
                    "source": r.get("source"),
                    "url": r.get("url"),
                }
            )
        return news_list

    except PsycopgError as e:
        logger.error(f"Error al obtener noticias de {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise
    finally:
        if conn is not None and not conn.closed:
            conn.close()


def _get_ml_predictions_performance(symbol: str, last_n_days: int = 7) -> Dict[str, Any]:
    """
    Obtiene métricas de rendimiento de los modelos ML validados.
    
    Calcula MAE y RMSE por modelo para las últimas N predicciones validadas,
    y devuelve el mejor modelo según MAE.
    
    Args:
        symbol: Símbolo del activo
        last_n_days: Número de días a considerar para las métricas
        
    Returns:
        Dict con métricas por modelo y mejor modelo
    """
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Obtener predicciones validadas (donde true_value no es NULL)
            cur.execute(
                """
                SELECT 
                    model_name,
                    AVG(error_abs) as avg_mae,
                    SQRT(AVG(POWER(error_abs, 2))) as rmse,
                    COUNT(*) as n_predictions,
                    AVG(predicted_value) as avg_predicted,
                    AVG(true_value) as avg_actual
                FROM ml_predictions
                WHERE symbol = %s
                  AND true_value IS NOT NULL
                  AND prediction_date >= CURRENT_DATE - INTERVAL '%s days'
                GROUP BY model_name
                ORDER BY avg_mae ASC;
                """,
                (symbol, last_n_days),
            )
            rows = cur.fetchall()

        if not rows:
            return {
                "models": [],
                "best_model": None,
                "message": f"No hay predicciones validadas en los últimos {last_n_days} días"
            }

        models_performance = []
        for r in rows:
            models_performance.append({
                "model_name": r["model_name"],
                "mae": float(r["avg_mae"]) if r["avg_mae"] else None,
                "rmse": float(r["rmse"]) if r["rmse"] else None,
                "n_predictions": int(r["n_predictions"]),
                "avg_predicted": float(r["avg_predicted"]) if r["avg_predicted"] else None,
                "avg_actual": float(r["avg_actual"]) if r["avg_actual"] else None,
            })

        # El mejor modelo es el primero (menor MAE)
        best_model = models_performance[0]["model_name"] if models_performance else None

        return {
            "models": models_performance,
            "best_model": best_model,
            "evaluation_period_days": last_n_days,
        }

    except PsycopgError as e:
        logger.error(f"Error al obtener métricas de predicciones de {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        return {
            "models": [],
            "best_model": None,
            "error": str(e)
        }
    finally:
        if conn is not None and not conn.closed:
            conn.close()


def _format_email_text(
    symbol: str,
    last_date,
    last_close,
    prev_close,
    abs_change,
    pct_change,
    indicators: Dict[str, Any],
    signals: Dict[str, Any],
    news: list,
) -> str:
    """
    Construye un texto resumen en castellano para usar en email.
    """
    if not last_date or last_close is None:
        return f"No hay datos suficientes para generar un resumen diario de {symbol}."

    fecha_str = last_date.strftime("%d/%m/%Y")
    linea_precio = f"Cierre de {symbol} el {fecha_str}: {last_close:,.2f} puntos"
    if abs_change is not None and pct_change is not None:
        signo = "+" if abs_change >= 0 else "-"
        linea_precio += f" ({signo}{abs(abs_change):,.2f}, {signo}{abs(pct_change):.2f}%)."
    else:
        linea_precio += " (sin referencia del día anterior)."

    # Señales
    s_simple = signals.get("simple")
    s_ensemble = signals.get("ensemble")

    def _interpreta(s):
        if s == 1:
            return "señal alcista (+1)"
        if s == -1:
            return "señal bajista (-1)"
        if s == 0:
            return "señal neutra (0)"
        return "sin señal disponible"

    linea_seniales = (
        f"Señal simple: {_interpreta(s_simple)}. "
        f"Señal ensemble: {_interpreta(s_ensemble)}."
    )

    # Indicadores
    sma20 = indicators.get("sma_20")
    sma50 = indicators.get("sma_50")
    rsi14 = indicators.get("rsi_14")
    vol20 = indicators.get("vol_20")

    partes_indicadores = []
    if sma20 is not None:
        partes_indicadores.append(f"SMA20 ≈ {sma20:,.2f}")
    if sma50 is not None:
        partes_indicadores.append(f"SMA50 ≈ {sma50:,.2f}")
    if rsi14 is not None:
        partes_indicadores.append(f"RSI14 ≈ {rsi14:.1f}")
    if vol20 is not None:
        partes_indicadores.append(f"Volatilidad 20 días ≈ {vol20:.4f}")

    if partes_indicadores:
        linea_indicadores = "Indicadores técnicos: " + ", ".join(partes_indicadores) + "."
    else:
        linea_indicadores = (
            "No hay indicadores técnicos suficientes calculados para esta fecha."
        )

    # Noticias
    if news:
        linea_news = "Noticias recientes:\n" + "\n".join(
            [f" - {n['title']}" for n in news]
        )
    else:
        linea_news = (
            "No hay noticias recientes registradas en la base de datos para este activo."
        )

    texto = (
        linea_precio
        + "\n\n"
        + linea_seniales
        + "\n\n"
        + linea_indicadores
        + "\n\n"
        + linea_news
    )
    return texto


def build_daily_summary(symbol: str = "^IBEX", include_ml_performance: bool = True) -> Dict[str, Any]:
    """
    Construye un resumen diario listo para que lo consuma n8n:
    - precios (último, anterior, variación)
    - indicadores del día
    - última señal simple y ensemble
    - últimas noticias
    - rendimiento de modelos ML (últimos 7 días)
    - texto plano para email
    
    Args:
        symbol: Símbolo del activo
        include_ml_performance: Si True, incluye métricas de predicciones ML
    """
    last_date, last_close, prev_close, abs_change, pct_change = _get_latest_price(symbol)
    indicators = _get_indicators_for_date(symbol, last_date)
    _, signals = _get_latest_signals(symbol)
    news = _get_recent_news(symbol, limit=5)
    
    # Obtener rendimiento de modelos ML
    ml_performance = None
    if include_ml_performance:
        ml_performance = _get_ml_predictions_performance(symbol, last_n_days=7)

    email_text = _format_email_text(
        symbol,
        last_date,
        last_close,
        prev_close,
        abs_change,
        pct_change,
        indicators,
        signals,
        news,
    )
    
    # Añadir info de ML al email si está disponible
    if ml_performance and ml_performance.get("best_model"):
        best = ml_performance["best_model"]
        best_mae = next((m["mae"] for m in ml_performance["models"] if m["model_name"] == best), None)
        email_text += f"\n\n📊 Mejor modelo ML (últimos 7 días): {best} (MAE: {best_mae:.2f} puntos)"

    summary: Dict[str, Any] = {
        "symbol": symbol,
        "date": last_date.isoformat() if last_date else None,
        "price": {
            "last": last_close,
            "prev": prev_close,
            "abs_change": abs_change,
            "pct_change": pct_change,
        },
        "indicators": indicators,
        "signals": signals,
        "news": news,
        "ml_performance": ml_performance,
        "email_text": email_text,
    }

    logger.info(f"Resumen diario construido para {symbol} en fecha {last_date}")
    return summary
