from datetime import date
from psycopg2 import Error as PsycopgError
from .config import get_db_conn


def save_daily_predictions(
    symbol: str,
    prediction_date: date,
    run_date: date,
    predictions: dict,
):
    """
    Guarda en la tabla ml_predictions las predicciones diarias de varios modelos.

    Parámetro predictions:
        dict con estructura:
        {
            "LinearRegression": {
                "price": 15920.52,
                "signal": -1,
            },
            "RandomForest": {
                "price": 15897.88,
                "signal": -1,
            },
            "ensemble": {
                "price": 15900.00,   # opcional, puedes poner None
                "signal": -1,
            },
            ...
        }
    """
    conn = None
    try:
        conn = get_db_conn()

        with conn.cursor() as cur:
            for model_name, values in predictions.items():
                predicted_price = values.get("price")      # puede ser float o None
                predicted_signal = values.get("signal")    # normalmente -1, 0, 1

                # INSERT con ON CONFLICT sobre la clave única
                cur.execute(
                    """
                    INSERT INTO ml_predictions (
                        symbol,
                        prediction_date,
                        run_date,
                        model_name,
                        predicted_value,
                        predicted_signal,
                        true_value,
                        error_abs
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL)
                    ON CONFLICT (symbol, prediction_date, model_name, run_date)
                    DO UPDATE SET
                        predicted_value = EXCLUDED.predicted_value,
                        predicted_signal = EXCLUDED.predicted_signal,
                        true_value = NULL,
                        error_abs = NULL;
                    """,
                    (
                        symbol,
                        prediction_date,
                        run_date,
                        model_name,
                        float(predicted_price) if predicted_price is not None else None,
                        int(predicted_signal) if predicted_signal is not None else None,
                    ),
                )

        conn.commit()

    except PsycopgError:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
