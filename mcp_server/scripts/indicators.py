from scripts.clean_data import fetch_and_clean_prices

def get_indicators(symbol: str = "^IBEX", days: int = 90):
    df = fetch_and_clean_prices(symbol, days)

    df["ema10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["momentum"] = df["Close"] - df["Close"].shift(5)
    df["volatility"] = df["ret"].rolling(window=10).std()

    df = df.dropna()
    return df
