import yfinance as yf
import os

os.makedirs("./data", exist_ok=True)
data = yf.download("^IBEX", period="2y")  # últimos 2 años
data.to_csv("./data/^IBEX_prices.csv")
print("CSV guardado en ./data/^IBEX_prices.csv")
