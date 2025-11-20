import pandas as pd

def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df.dropna(subset=["Close"])
    # ejemplo: rellenar festivos con forward fill si lo necesitas
    df = df.asfreq("B")  # días hábiles
    df["Close"] = df["Close"].ffill()
    # añadir retornos
    df["return_1d"] = df["Close"].pct_change()
    return df