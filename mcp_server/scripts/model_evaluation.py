"""Módulo de evaluación de modelos ML basado en predicciones validadas.

Analiza el rendimiento histórico de los modelos ML para ayudar en decisiones
de reentrenamiento y selección de modelos.
"""

from datetime import date, timedelta
from typing import Dict, List, Any
from psycopg2 import Error as PsycopgError
from .config import get_db_conn
from . import logger


def get_model_performance_report(
    symbol: str,
    start_date: date = None,
    end_date: date = None,
    min_predictions: int = 3
) -> Dict[str, Any]:
    """
    Genera un reporte completo de rendimiento de modelos ML.
    
    Analiza predicciones validadas para determinar:
    - MAE y RMSE por modelo
    - Accuracy de señales (buy/sell/hold)
    - Modelos que necesitan reentrenamiento
    - Mejor modelo actual
    
    Args:
        symbol: Símbolo del activo
        start_date: Fecha inicio del análisis (por defecto: 30 días atrás)
        end_date: Fecha fin del análisis (por defecto: hoy)
        min_predictions: Mínimo de predicciones validadas para considerar el modelo
        
    Returns:
        Dict con análisis completo de rendimiento
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=30)
    if end_date is None:
        end_date = date.today()
    
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Obtener métricas agregadas por modelo
            cur.execute(
                """
                SELECT 
                    model_name,
                    COUNT(*) as n_predictions,
                    AVG(error_abs) as avg_mae,
                    SQRT(AVG(POWER(error_abs, 2))) as rmse,
                    MIN(error_abs) as best_prediction,
                    MAX(error_abs) as worst_prediction,
                    AVG(predicted_value) as avg_predicted,
                    AVG(true_value) as avg_actual,
                    STDDEV(error_abs) as std_error
                FROM ml_predictions
                WHERE symbol = %s
                  AND true_value IS NOT NULL
                  AND prediction_date BETWEEN %s AND %s
                GROUP BY model_name
                HAVING COUNT(*) >= %s
                ORDER BY avg_mae ASC;
                """,
                (symbol, start_date, end_date, min_predictions),
            )
            performance_rows = cur.fetchall()
            
            # Obtener accuracy de señales (si predicted_signal existe)
            cur.execute(
                """
                SELECT 
                    model_name,
                    COUNT(CASE WHEN predicted_signal = 1 
                           AND predicted_value < true_value THEN 1 END) as correct_buys,
                    COUNT(CASE WHEN predicted_signal = 1 THEN 1 END) as total_buys,
                    COUNT(CASE WHEN predicted_signal = -1 
                           AND predicted_value > true_value THEN 1 END) as correct_sells,
                    COUNT(CASE WHEN predicted_signal = -1 THEN 1 END) as total_sells,
                    COUNT(*) as total_predictions
                FROM ml_predictions
                WHERE symbol = %s
                  AND true_value IS NOT NULL
                  AND predicted_signal IS NOT NULL
                  AND prediction_date BETWEEN %s AND %s
                GROUP BY model_name;
                """,
                (symbol, start_date, end_date),
            )
            signal_rows = cur.fetchall()
        
        # Procesar resultados
        models_analysis = []
        signal_accuracy = {row["model_name"]: row for row in signal_rows}
        
        for row in performance_rows:
            model_name = row["model_name"]
            
            # Calcular signal accuracy si existe
            signal_data = signal_accuracy.get(model_name, {})
            buy_accuracy = None
            sell_accuracy = None
            
            if signal_data:
                total_buys = signal_data.get("total_buys", 0)
                total_sells = signal_data.get("total_sells", 0)
                
                if total_buys > 0:
                    buy_accuracy = (signal_data.get("correct_buys", 0) / total_buys) * 100
                if total_sells > 0:
                    sell_accuracy = (signal_data.get("correct_sells", 0) / total_sells) * 100
            
            # Determinar si necesita reentrenamiento
            needs_retrain = False
            retrain_reason = []
            
            if row["avg_mae"] and row["avg_mae"] > 200:  # MAE muy alto
                needs_retrain = True
                retrain_reason.append("MAE alto")
            
            if row["std_error"] and row["std_error"] > 150:  # Errores inconsistentes
                needs_retrain = True
                retrain_reason.append("Errores inconsistentes")
            
            if buy_accuracy is not None and buy_accuracy < 40:  # Baja precisión
                needs_retrain = True
                retrain_reason.append("Baja accuracy en señales de compra")
            
            models_analysis.append({
                "model_name": model_name,
                "n_predictions": int(row["n_predictions"]),
                "mae": float(row["avg_mae"]) if row["avg_mae"] else None,
                "rmse": float(row["rmse"]) if row["rmse"] else None,
                "best_prediction_error": float(row["best_prediction"]) if row["best_prediction"] else None,
                "worst_prediction_error": float(row["worst_prediction"]) if row["worst_prediction"] else None,
                "std_error": float(row["std_error"]) if row["std_error"] else None,
                "buy_signal_accuracy": buy_accuracy,
                "sell_signal_accuracy": sell_accuracy,
                "needs_retrain": needs_retrain,
                "retrain_reasons": retrain_reason if retrain_reason else None,
            })
        
        # Identificar mejor modelo
        best_model = models_analysis[0]["model_name"] if models_analysis else None
        
        # Modelos que necesitan reentrenamiento
        models_to_retrain = [m["model_name"] for m in models_analysis if m["needs_retrain"]]
        
        logger.info(
            f"Reporte de rendimiento para {symbol}: "
            f"{len(models_analysis)} modelos evaluados, "
            f"mejor={best_model}, "
            f"necesitan reentrenamiento={len(models_to_retrain)}"
        )
        
        return {
            "symbol": symbol,
            "evaluation_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": (end_date - start_date).days + 1,
            },
            "models": models_analysis,
            "best_model": best_model,
            "models_to_retrain": models_to_retrain,
            "summary": {
                "total_models": len(models_analysis),
                "avg_mae_all_models": sum(m["mae"] for m in models_analysis if m["mae"]) / len(models_analysis) if models_analysis else None,
                "models_needing_retrain": len(models_to_retrain),
            }
        }
    
    except PsycopgError as e:
        logger.error(f"Error al generar reporte de rendimiento para {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        return {
            "symbol": symbol,
            "error": "database_error",
            "message": str(e),
            "models": [],
        }
    finally:
        if conn is not None and not conn.closed:
            conn.close()


def should_retrain_models(symbol: str, mae_threshold: float = 200.0) -> Dict[str, Any]:
    """
    Determina si los modelos necesitan ser reentrenados.
    
    Analiza los últimos 7 días de predicciones validadas y recomienda
    reentrenamiento si el rendimiento ha caído.
    
    Args:
        symbol: Símbolo del activo
        mae_threshold: Umbral de MAE para recomendar reentrenamiento
        
    Returns:
        Dict con recomendación de reentrenamiento
    """
    today = date.today()
    week_ago = today - timedelta(days=7)
    
    report = get_model_performance_report(
        symbol=symbol,
        start_date=week_ago,
        end_date=today,
        min_predictions=2  # Al menos 2 predicciones en la semana
    )
    
    if not report.get("models"):
        return {
            "symbol": symbol,
            "should_retrain": False,
            "reason": "No hay suficientes predicciones validadas en los últimos 7 días",
            "recommendation": "Esperar más datos antes de reentrenar",
        }
    
    # Verificar si algún modelo tiene MAE por encima del umbral
    avg_mae = report["summary"].get("avg_mae_all_models")
    models_to_retrain = report.get("models_to_retrain", [])
    
    should_retrain = False
    reasons = []
    
    if avg_mae and avg_mae > mae_threshold:
        should_retrain = True
        reasons.append(f"MAE promedio ({avg_mae:.2f}) supera el umbral ({mae_threshold})")
    
    if len(models_to_retrain) >= 3:  # Mayoría de modelos necesitan reentrenamiento
        should_retrain = True
        reasons.append(f"{len(models_to_retrain)} modelos con bajo rendimiento")
    
    return {
        "symbol": symbol,
        "should_retrain": should_retrain,
        "reasons": reasons if reasons else ["Rendimiento aceptable"],
        "avg_mae": avg_mae,
        "models_to_retrain": models_to_retrain,
        "recommendation": "Reentrenar modelos inmediatamente" if should_retrain else "Continuar con modelos actuales",
        "detailed_report": report,
    }
