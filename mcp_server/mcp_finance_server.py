from fastapi import FastAPI
from scripts.clean_data import fetch_and_clean_prices
from scripts.indicators import get_indicators
from scripts.models import predict_simple, predict_ensemble

app = FastAPI()

@app.get("/indicadores")
def indicadores(symbol: str = "^IBEX", days: int = 90):
    df = get_indicators(symbol=symbol, days=days)
    last_row = df.iloc[-1].to_dict()
    return {"last": last_row, "rows": len(df)}

@app.get("/predecir_simple")
def predecir_simple(symbol: str = "^IBEX", days: int = 90):
    df = fetch_and_clean_prices(symbol, days)
    pred = predict_simple(df)
    return pred

@app.get("/predecir_ensemble")
def predecir_ensemble(symbol: str = "^IBEX", days: int = 90):
    # 🟢 Cambiado: ahora usamos los datos con indicadores completos
    df = get_indicators(symbol=symbol, days=days)
    pred = predict_ensemble(df)
    return pred
