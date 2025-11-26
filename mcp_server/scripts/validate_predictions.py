from datetime import date, timedelta
from psycopg2 import Error as PsycopgError
from .config import get_db_conn


def validate_predictions_for_date(target_date: date):
    """
    Valida las predicciones de ml_predictions para una fecha dada:
    - Busca el precio real de cierre en la tabla prices.
    - Actualiza true_value y error_abs = |predicted_value - true_value|.
    """
    conn = None
    try:
        conn = get_db_conn()
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
            updated = 0

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

        conn.commit()
        return {
            "target_date": target_date.isoformat(),
            "symbols_with_price": list(real_prices.keys()),
            "rows_updated": updated,
        }

    except PsycopgError:
        if conn:
            conn.rollback()
        raise

    finally:
        if conn:
            conn.close()


def validate_predictions_yesterday():
    today = date.today()
    target_date = today - timedelta(days=1)
    return validate_predictions_for_date(target_date)

