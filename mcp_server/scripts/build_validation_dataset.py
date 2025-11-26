import pandas as pd
from .config import get_db_conn  # si va en paquete, sería from .config import get_db_conn


def load_prices(conn):
    """
    Carga precios en un DataFrame: [symbol, date, close]
    """
    query = """
        SELECT symbol, date, close
        FROM prices
        ORDER BY symbol, date
    """
    return pd.read_sql(query, conn)


def load_predictions(conn):
    """
    Carga predicciones validadas en un DataFrame:
    [symbol, prediction_date, model_name,
     predicted_value, predicted_signal, true_value]
    """
    query = """
        SELECT
            symbol,
            prediction_date,
            model_name,
            predicted_value,
            predicted_signal,
            true_value
        FROM ml_predictions
        WHERE true_value IS NOT NULL
        ORDER BY symbol, prediction_date, model_name
    """
    return pd.read_sql(query, conn)


def build_validation_dataset():
    """
    Devuelve un DataFrame con:
    - prediction_date
    - symbol
    - model_name
    - predicted_value (precio predicho)
    - predicted_signal (señal del modelo)
    - true_value (precio real)
    - close_prev (cierre del día anterior)
    - real_return
    - direction_real (+1/-1/0)
    - direction_pred (= predicted_signal)
    - acierto (True si direction_pred == direction_real)
    """
    conn = get_db_conn()
    try:
        prices = load_prices(conn)
        preds = load_predictions(conn)
    finally:
        conn.close()

    # 1) Ordenar precios y calcular cierre del día anterior por símbolo
    prices = prices.sort_values(["symbol", "date"])
    prices["close_prev"] = prices.groupby("symbol")["close"].shift(1)

    # 2) Unir predicciones (prediction_date) con precios (date)
    df = preds.merge(
        prices[["symbol", "date", "close", "close_prev"]],
        left_on=["symbol", "prediction_date"],
        right_on=["symbol", "date"],
        how="left",
    )

    df.rename(columns={"date": "price_date"}, inplace=True)

    # 3) Calcular retorno real respecto al día anterior
    df["real_return"] = (df["true_value"] - df["close_prev"]) / df["close_prev"]

    # 4) Dirección real a partir del retorno
    threshold = 0.0  # sube si >0, baja si <0 (puedes subirlo si quieres umbral)

    def direction_from_return(r):
        if pd.isna(r):
            return None
        if r > threshold:
            return 1
        elif r < -threshold:
            return -1
        else:
            return 0

    df["direction_real"] = df["real_return"].apply(direction_from_return)

    # 5) Dirección predicha: usamos directamente la señal del modelo
    df["direction_pred"] = df["predicted_signal"]

    # 6) Acierto si coinciden las direcciones
    df["acierto"] = df["direction_real"] == df["direction_pred"]

    return df


if __name__ == "__main__":
    df = build_validation_dataset()
    df.to_csv("validation_dataset.csv", index=False)
    print("Guardado validation_dataset.csv con", len(df), "filas")
