
"""
Script de backfill de predicciones históricas.

PROBLEMAS CONOCIDOS:
1. ⚠️ LOOK-AHEAD BIAS: predict_ensemble() usa TODOS los datos hasta hoy,
   no solo los datos disponibles hasta la fecha del backfill.
   
   Esto hace que las predicciones "históricas" tengan información del futuro,
   invalidando cualquier análisis de rendimiento histórico.

2. ⚠️ EJECUCIÓN: Este script debe ejecutarse DENTRO del contenedor Docker:
   
   docker exec -it mcp_finance python -m scripts.backfill_predictions
   
   No funcionará si se ejecuta directamente desde tu máquina (DB_HOST=db no existe).

SOLUCIÓN RECOMENDADA:
- Modificar predict_ensemble() para aceptar un parámetro 'as_of_date'
- Filtrar datos en _load_features() para usar solo datos <= as_of_date
- Esto requiere refactorización significativa del sistema de modelos

POR AHORA: Usar solo para pruebas, NO para análisis de rendimiento real.
"""

from datetime import date, timedelta
from .config import get_db_conn           # o from .config import get_db_conn si usas paquete
from .save_predictions import save_daily_predictions
from .models import predict_ensemble      # o from .models import predict_ensemble
import psycopg2


def get_available_dates(symbol: str):
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date
                FROM prices
                WHERE symbol = %s
                ORDER BY date
                """,
                (symbol,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    # Si rows son dicts: [{'date': ...}, ...]
    return [r["date"] for r in rows]


def backfill_predictions_for_symbol(symbol: str, start_date: date = None, end_date: date = None):
    """
    Recorre un rango histórico de fechas y genera predicciones diarias
    como si se hubieran hecho en tiempo real, guardándolas en ml_predictions.

    Para cada fecha D en el rango:
      - Usamos predict_ensemble(symbol) para obtener la predicción para D (o D+1 según tu lógica).
      - Guardamos en ml_predictions con prediction_date = D (o D+1).
    """
    all_dates = get_available_dates(symbol)
    if not all_dates:
        print(f"No hay precios en 'prices' para {symbol}")
        return

    # Si no se especifica rango, usamos todo
    first = all_dates[0]
    last = all_dates[-1]

    if start_date is None:
        start_date = first
    if end_date is None:
        end_date = last

    # Filtrar fechas al rango deseado
    dates = [d for d in all_dates if start_date <= d <= end_date]

    print(f"Backfill para {symbol} desde {start_date} hasta {end_date} ({len(dates)} días)")

    for d in dates:
        # Aquí hay una decisión:
        # - d = fecha para la que quieres tener EL PRECIO REAL en prices.
        # - prediction_date podría ser d (predecir el cierre de ese día usando info de días anteriores)
        #   o d+1 (si tus modelos están definidos como 'predecir el siguiente día').
        #
        # En este ejemplo consideramos que prediction_date = d,
        # y que en producción llamarías a predict_ensemble al final de d-1.
        prediction_date = d
        run_date = d  # en backtest es irrelevante; puedes dejarlo igual

        # TODO: idealmente, aquí querrías una versión de predict_ensemble
        # que reciba 'as_of_date=prediction_date' y solo use datos hasta ese día.

        result = predict_ensemble(symbol)

        # Construir predictions_dict igual que en el endpoint /predecir_ensemble
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

        if "signal_ensemble" in result:
            prices = [
                m.get("prediction_next_day")
                for m in result.get("ml_models", [])
                if m.get("prediction_next_day") is not None
            ]
            avg_price = sum(prices) / len(prices) if prices else None

            predictions_dict["ensemble"] = {
                "price": avg_price,
                "signal": result["signal_ensemble"],
            }

        if predictions_dict:
            save_daily_predictions(
                symbol=symbol,
                prediction_date=prediction_date,
                run_date=run_date,
                predictions=predictions_dict,
            )
            print(f"[{prediction_date}] guardadas predicciones para {symbol}")
        else:
            print(f"[{prediction_date}] SIN predicciones para {symbol}")


if __name__ == "__main__":
    """
    Para ejecutar este script:
    
    1. Desde dentro del contenedor:
       docker exec -it mcp_finance python -m scripts.backfill_predictions
    
    2. O entrar al contenedor interactivamente:
       docker exec -it mcp_finance bash
       cd /app
       python -m scripts.backfill_predictions
    """
    # Ejemplo: backtest completo para ^IBEX
    symbol = "^IBEX"
    start = date(2025, 11, 17)
    end   = date(2025, 11, 21)
    
    print("⚠️  ADVERTENCIA: Este backfill tiene look-ahead bias!")
    print("   Las predicciones usan información del futuro.\n")
    
    backfill_predictions_for_symbol(symbol, start_date=start, end_date=end)
