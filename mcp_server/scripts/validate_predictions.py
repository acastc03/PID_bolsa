"""Módulo de validación de predicciones ML.

Compara las predicciones guardadas con los valores reales observados
y calcula métricas de error para evaluar el rendimiento de los modelos.
"""

from datetime import date, timedelta
from psycopg2 import Error as PsycopgError
from .config import get_db_conn


def validate_predictions_for_date(target_date: date):
    """Valida predicciones contra valores reales para una fecha específica.
    
    Workflow:
    1. Obtiene precios reales de cierre para la fecha objetivo
    2. Actualiza ml_predictions con true_value
    3. Calcula error_abs = |predicted_value - true_value|
    
    Args:
        target_date: Fecha para validar (debe existir en tabla prices)
        
    Returns:
        dict: {
            "target_date": "2025-11-26",
            "symbols_with_price": ["^IBEX", "^GSPC"],
            "rows_updated": 14  # Número de predicciones validadas
        }
        
    Raises:
        PsycopgError: Si hay error en la actualización
        
    Note:
        - Solo actualiza predicciones donde existe precio real
        - Útil para ejecutar diariamente y evaluar accuracy
        - Permite calcular MAE, RMSE por modelo posteriormente
    """
    conn = get_db_conn()
    real_prices = {}
    updated = 0

    try:
        with conn.cursor() as cur:
            # 1) Obtener precios reales de ese día
            cur.execute(
                """
                SELECT symbol, close
                FROM prices
                WHERE date = %s
                """,
                (target_date,),
            )
            rows = cur.fetchall()

            real_prices = {symbol: close for (symbol, close) in rows}

            # Si no hay precios para esa fecha, devolvemos algo informativo
            if not real_prices:
                return {
                    "target_date": target_date.isoformat(),
                    "symbols_with_price": [],
                    "rows_updated": 0,
                    "message": "No hay precios en 'prices' para esa fecha",
                }

            # 2) Actualizar ml_predictions para cada símbolo con precio real
            for symbol, real_price in real_prices.items():
                cur.execute(
                    """
                    UPDATE ml_predictions
                    SET
                        true_value = %s,
                        error_abs = ABS(predicted_value - %s)
                    WHERE prediction_date = %s
                      AND symbol = %s;
                    """,
                    (real_price, real_price, target_date, symbol),
                )
                updated += cur.rowcount

        # Si todo ha ido bien, confirmamos
        conn.commit()

        return {
            "target_date": target_date.isoformat(),
            "symbols_with_price": list(real_prices.keys()),
            "rows_updated": updated,
        }

    except PsycopgError as e:
        if conn is not None and not conn.closed:
            conn.rollback()
        print(f"[validate_predictions_for_date] Error de BD: {e}")
        # ⚠️ En vez de `raise`, devolvemos un error controlado
        return {
            "target_date": target_date.isoformat(),
            "symbols_with_price": [],
            "rows_updated": 0,
            "error": "database_error",
            "message": "Error de base de datos al validar predicciones",
        }

    finally:
        # Cerramos la conexión solo si sigue abierta
        if conn is not None and not conn.closed:
            conn.close()


def validate_predictions_yesterday():
    """Valida las predicciones del día anterior (ayer).
    
    Atajo conveniente para validación automática diaria.
    Típicamente se ejecuta cada mañana para validar las predicciones
    del día anterior una vez que los precios reales están disponibles.
    
    Returns:
        dict: Resultado de validate_predictions_for_date para ayer
        
    Example:
        En n8n, programar a las 9:00 AM:
        POST /validate_predictions
        
        Esto validará las predicciones del día anterior automáticamente.
    """
    today = date.today()
    target_date = today - timedelta(days=1)
    return validate_predictions_for_date(target_date)
