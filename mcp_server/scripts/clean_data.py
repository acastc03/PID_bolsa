import pandas as pd
import os

def fetch_and_clean_prices(symbol: str = "^IBEX", days: int = 90):
    cache_file = f"./data/{symbol}_prices.csv"
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"No existe el CSV cacheado {cache_file}. Crea el archivo manualmente para continuar."
        )

    # Cargar CSV con estructura: Date, Close, High, Low, Open, Volume
    df = pd.read_csv(cache_file, skiprows=3)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Adj Close"] = df["Close"]

    # Features básicos
    df["ret"] = df["Close"].pct_change()
    df["sma20"] = df["Close"].rolling(window=20).mean()
    df["sma50"] = df["Close"].rolling(window=50).mean()

    df = df.dropna()
    return df
