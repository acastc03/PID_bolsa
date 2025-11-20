# mcp_server/scripts/models.py

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

from .config import get_db_conn
from . import logger
from .model_storage import save_model, load_model, model_exists


def _load_features(symbol: str) -> pd.DataFrame:
    """
    Carga precios + indicadores para un símbolo y construye un DataFrame de features
    indexado por fecha.
    """
    conn = get_db_conn()
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                p.date,
                p.close,
                i.sma_20,
                i.sma_50,
                i.vol_20,
                i.rsi_14
            FROM prices p
            LEFT JOIN indicators i
              ON p.symbol = i.symbol
             AND p.date = i.date
            WHERE p.symbol = %s
            ORDER BY p.date
            """,
            (symbol,),
        )
        rows = cur.fetchall()

    if not rows:
        logger.warning(f"No hay datos de precios/indicadores para {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    
    # Añadir features adicionales para los modelos ML
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["momentum"] = df["close"].diff(5)
    df["volatility"] = df["close"].rolling(window=20).std()
    
    return df


def evaluate_model(y_true, y_pred):
    """Calcula MAE y RMSE para evaluar modelos."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


# ========================================================================
# REGLAS BASADAS EN INDICADORES
# ========================================================================

def _rule_based_signal(row: pd.Series) -> int:
    """
    Modelo simple basado en reglas:
    +1 → cierre por encima de SMA20 y RSI entre 40 y 70
    -1 → cierre por debajo de SMA20 y RSI entre 30 y 60
     0 → resto (neutral o sobrecompra/sobreventa)
    """
    close = row["close"]
    sma20 = row["sma_20"]
    rsi = row["rsi_14"]

    if pd.isna(close) or pd.isna(sma20) or pd.isna(rsi):
        return 0

    if (close > sma20) and (40 <= rsi <= 70):
        return 1
    if (close < sma20) and (30 <= rsi <= 60):
        return -1
    return 0


def _rule_based_signal_alt(row: pd.Series) -> int:
    """
    Segunda variante: tiene en cuenta volatilidad y RSI.
    +1 → close > sma20 y vol_20 baja y RSI < 65
    -1 → close < sma20 y vol_20 alta o RSI > 75
     0 → resto
    """
    close = row["close"]
    sma20 = row["sma_20"]
    vol20 = row["vol_20"]
    rsi = row["rsi_14"]

    if pd.isna(close) or pd.isna(sma20) or pd.isna(rsi):
        return 0

    if pd.isna(vol20):
        vol20 = 0.01

    if (close > sma20) and (vol20 < 0.01) and (rsi < 65):
        return 1
    if (close < sma20) and ((vol20 > 0.015) or (rsi > 75)):
        return -1
    return 0


def _rule_based_signal_contrarian(row: pd.Series) -> int:
    """
    Tercera variante 'contrarian':
    +1 → RSI < 30 (sobreventa)
    -1 → RSI > 70 (sobrecompra)
     0 → resto
    """
    rsi = row["rsi_14"]
    if pd.isna(rsi):
        return 0
    if rsi < 30:
        return 1
    if rsi > 70:
        return -1
    return 0


# ========================================================================
# MODELOS DE MACHINE LEARNING CON PERSISTENCIA
# ========================================================================

def _predict_ml_models(df: pd.DataFrame, symbol: str = "^IBEX", force_retrain: bool = False) -> list:
    """
    Entrena y predice con 7 modelos de ML.
    - Si force_retrain=False, intenta cargar modelos guardados
    - Si no existen, entrena nuevos y los guarda automáticamente
    
    Args:
        df: DataFrame con features
        symbol: Símbolo del activo
        force_retrain: Si True, fuerza reentrenamiento aunque existan modelos
    
    Returns:
        Lista de resultados de cada modelo
    """
    results = []
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Preparar features (eliminar NaN)
    df_clean = df.dropna()
    if len(df_clean) < 50:
        logger.warning("No hay suficientes datos para entrenar modelos ML")
        return results
    
    # Features: asegurarnos de que existen
    required_features = ["sma_20", "sma_50", "ema_10", "ema_50", "momentum", "volatility"]
    if not all(col in df_clean.columns for col in required_features):
        logger.warning("Faltan features requeridas para modelos ML")
        return results
    
    X = df_clean[required_features]
    y = df_clean["close"]
    
    # Split: usar todos menos el último para entrenar, último para predecir
    X_train, X_test = X[:-1], X[-1:]
    y_train, y_test = y[:-1], y[-1:]
    
    current_price = y_test.iloc[0]

    # 1️⃣ Linear Regression
    model_name = "LinearRegression"
    try:
        if not force_retrain and model_exists(symbol, model_name):
            logger.info(f"📦 Usando modelo guardado: {model_name}")
            model_data = load_model(symbol, model_name)
            if model_data:
                model = model_data["model"]
                pred = model.predict(X_test)[0]
                
                results.append({
                    "model_name": model_name,
                    "prediction_next_day": float(pred),
                    "signal_next_day": 1 if pred > current_price else -1,
                    "MAE": model_data["metadata"].get("MAE", 0),
                    "RMSE": model_data["metadata"].get("RMSE", 0),
                    "from_cache": True,
                    "training_date": model_data.get("training_date")
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            lr = LinearRegression().fit(X_train, y_train)
            pred = lr.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, lr.predict(X_train))
            
            metadata = {"MAE": float(mae), "RMSE": float(rmse), "n_samples": len(X_train)}
            save_model(lr, symbol, model_name, today, metadata)
            
            results.append({
                "model_name": model_name,
                "prediction_next_day": float(pred),
                "signal_next_day": 1 if pred > current_price else -1,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "from_cache": False,
                "training_date": today
            })
    except Exception as e:
        logger.error(f"Error en {model_name}: {e}")

    # 2️⃣ Random Forest
    model_name = "RandomForest"
    try:
        if not force_retrain and model_exists(symbol, model_name):
            logger.info(f"📦 Usando modelo guardado: {model_name}")
            model_data = load_model(symbol, model_name)
            if model_data:
                model = model_data["model"]
                pred = model.predict(X_test)[0]
                
                results.append({
                    "model_name": model_name,
                    "prediction_next_day": float(pred),
                    "signal_next_day": 1 if pred > current_price else -1,
                    "MAE": model_data["metadata"].get("MAE", 0),
                    "RMSE": model_data["metadata"].get("RMSE", 0),
                    "from_cache": True,
                    "training_date": model_data.get("training_date")
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
            pred = rf.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, rf.predict(X_train))
            
            metadata = {"MAE": float(mae), "RMSE": float(rmse), "n_samples": len(X_train)}
            save_model(rf, symbol, model_name, today, metadata)
            
            results.append({
                "model_name": model_name,
                "prediction_next_day": float(pred),
                "signal_next_day": 1 if pred > current_price else -1,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "from_cache": False,
                "training_date": today
            })
    except Exception as e:
        logger.error(f"Error en {model_name}: {e}")

    # 3️⃣ Prophet
    model_name = "Prophet"
    try:
        if not force_retrain and model_exists(symbol, model_name):
            logger.info(f"📦 Usando modelo guardado: {model_name}")
            model_data = load_model(symbol, model_name)
            if model_data:
                model = model_data["model"]
                future = model.make_future_dataframe(periods=1)
                forecast = model.predict(future)
                prophet_pred = forecast["yhat"].iloc[-1]
                
                results.append({
                    "model_name": model_name,
                    "prediction_next_day": float(prophet_pred),
                    "signal_next_day": 1 if prophet_pred > current_price else -1,
                    "MAE": model_data["metadata"].get("MAE", 0),
                    "RMSE": model_data["metadata"].get("RMSE", 0),
                    "from_cache": True,
                    "training_date": model_data.get("training_date")
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            prophet_df = pd.DataFrame({"ds": df_clean.index, "y": df_clean["close"]})
            prophet = Prophet(daily_seasonality=True)
            prophet.fit(prophet_df)
            
            future = prophet.make_future_dataframe(periods=1)
            forecast = prophet.predict(future)
            prophet_pred = forecast["yhat"].iloc[-1]
            prophet_mae, prophet_rmse = evaluate_model(prophet_df["y"], forecast["yhat"][:-1])
            
            metadata = {"MAE": float(prophet_mae), "RMSE": float(prophet_rmse), "n_samples": len(prophet_df)}
            save_model(prophet, symbol, model_name, today, metadata)
            
            results.append({
                "model_name": model_name,
                "prediction_next_day": float(prophet_pred),
                "signal_next_day": 1 if prophet_pred > current_price else -1,
                "MAE": float(prophet_mae),
                "RMSE": float(prophet_rmse),
                "from_cache": False,
                "training_date": today
            })
    except Exception as e:
        logger.error(f"Error en {model_name}: {e}")

    # 4️⃣ XGBoost
    model_name = "XGBoost"
    try:
        if not force_retrain and model_exists(symbol, model_name):
            logger.info(f"📦 Usando modelo guardado: {model_name}")
            model_data = load_model(symbol, model_name)
            if model_data:
                model = model_data["model"]
                pred = model.predict(X_test)[0]
                
                results.append({
                    "model_name": model_name,
                    "prediction_next_day": float(pred),
                    "signal_next_day": 1 if pred > current_price else -1,
                    "MAE": model_data["metadata"].get("MAE", 0),
                    "RMSE": model_data["metadata"].get("RMSE", 0),
                    "from_cache": True,
                    "training_date": model_data.get("training_date")
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
            xgb.fit(X_train, y_train)
            pred = xgb.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, xgb.predict(X_train))
            
            metadata = {"MAE": float(mae), "RMSE": float(rmse), "n_samples": len(X_train)}
            save_model(xgb, symbol, model_name, today, metadata)
            
            results.append({
                "model_name": model_name,
                "prediction_next_day": float(pred),
                "signal_next_day": 1 if pred > current_price else -1,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "from_cache": False,
                "training_date": today
            })
    except Exception as e:
        logger.error(f"Error en {model_name}: {e}")

    # 5️⃣ SVR
    model_name = "SVR"
    try:
        if not force_retrain and model_exists(symbol, model_name):
            logger.info(f"📦 Usando modelo guardado: {model_name}")
            model_data = load_model(symbol, model_name)
            if model_data:
                model = model_data["model"]
                pred = model.predict(X_test)[0]
                
                results.append({
                    "model_name": model_name,
                    "prediction_next_day": float(pred),
                    "signal_next_day": 1 if pred > current_price else -1,
                    "MAE": model_data["metadata"].get("MAE", 0),
                    "RMSE": model_data["metadata"].get("RMSE", 0),
                    "from_cache": True,
                    "training_date": model_data.get("training_date")
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
            svr.fit(X_train, y_train)
            pred = svr.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, svr.predict(X_train))
            
            metadata = {"MAE": float(mae), "RMSE": float(rmse), "n_samples": len(X_train)}
            save_model(svr, symbol, model_name, today, metadata)
            
            results.append({
                "model_name": model_name,
                "prediction_next_day": float(pred),
                "signal_next_day": 1 if pred > current_price else -1,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "from_cache": False,
                "training_date": today
            })
    except Exception as e:
        logger.error(f"Error en {model_name}: {e}")

    # 6️⃣ LightGBM
    model_name = "LightGBM"
    try:
        if not force_retrain and model_exists(symbol, model_name):
            logger.info(f"📦 Usando modelo guardado: {model_name}")
            model_data = load_model(symbol, model_name)
            if model_data:
                model = model_data["model"]
                pred = model.predict(X_test)[0]
                
                results.append({
                    "model_name": model_name,
                    "prediction_next_day": float(pred),
                    "signal_next_day": 1 if pred > current_price else -1,
                    "MAE": model_data["metadata"].get("MAE", 0),
                    "RMSE": model_data["metadata"].get("RMSE", 0),
                    "from_cache": True,
                    "training_date": model_data.get("training_date")
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            lgbm = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            lgbm.fit(X_train, y_train)
            pred = lgbm.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, lgbm.predict(X_train))
            
            metadata = {"MAE": float(mae), "RMSE": float(rmse), "n_samples": len(X_train)}
            save_model(lgbm, symbol, model_name, today, metadata)
            
            results.append({
                "model_name": model_name,
                "prediction_next_day": float(pred),
                "signal_next_day": 1 if pred > current_price else -1,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "from_cache": False,
                "training_date": today
            })
    except Exception as e:
        logger.error(f"Error en {model_name}: {e}")

    # 7️⃣ CatBoost
    model_name = "CatBoost"
    try:
        if not force_retrain and model_exists(symbol, model_name):
            logger.info(f"📦 Usando modelo guardado: {model_name}")
            model_data = load_model(symbol, model_name)
            if model_data:
                model = model_data["model"]
                pred = model.predict(X_test)[0]
                
                results.append({
                    "model_name": model_name,
                    "prediction_next_day": float(pred),
                    "signal_next_day": 1 if pred > current_price else -1,
                    "MAE": model_data["metadata"].get("MAE", 0),
                    "RMSE": model_data["metadata"].get("RMSE", 0),
                    "from_cache": True,
                    "training_date": model_data.get("training_date")
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            cat = CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                silent=True,
                random_state=42
            )
            cat.fit(X_train, y_train)
            pred = cat.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, cat.predict(X_train))
            
            metadata = {"MAE": float(mae), "RMSE": float(rmse), "n_samples": len(X_train)}
            save_model(cat, symbol, model_name, today, metadata)
            
            results.append({
                "model_name": model_name,
                "prediction_next_day": float(pred),
                "signal_next_day": 1 if pred > current_price else -1,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "from_cache": False,
                "training_date": today
            })
    except Exception as e:
        logger.error(f"Error en {model_name}: {e}")

    return results


# ========================================================================
# FUNCIONES PÚBLICAS
# ========================================================================

def predict_simple(symbol: str) -> int:
    """
    Devuelve la señal simple (+1, 0, -1) para la última fecha disponible
    usando reglas basadas en indicadores.
    """
    df = _load_features(symbol)
    if df.empty:
        return 0

    last_row = df.iloc[-1]
    sig = _rule_based_signal(last_row)
    logger.info(f"Señal simple para {symbol} en {df.index[-1].date()}: {sig}")
    return int(sig)


def predict_ensemble(symbol: str, force_retrain: bool = False) -> dict:
    """
    Calcula señales con:
    - 3 reglas basadas en indicadores (solo como referencia)
    - 7 modelos de ML (estos son los que votan)
    
    El ensemble se calcula SOLO con los modelos ML.
    
    Args:
        symbol: Símbolo del activo
        force_retrain: Si True, fuerza reentrenamiento de todos los modelos
    
    Devuelve:
        - rule_signals: señales de las 3 reglas (informativo)
        - ml_models: lista de resultados de los 7 modelos ML
        - signal_ensemble: señal final por votación (SOLO modelos ML)
    """
    df = _load_features(symbol)
    if df.empty:
        return {
            "rule_signals": [],
            "ml_models": [],
            "signal_ensemble": 0
        }

    last_row = df.iloc[-1]

    # Señales basadas en reglas (solo informativas, NO votan)
    s1 = _rule_based_signal(last_row)
    s2 = _rule_based_signal_alt(last_row)
    s3 = _rule_based_signal_contrarian(last_row)
    rule_signals = [s1, s2, s3]

    # Señales de modelos ML (estos SÍ votan)
    ml_results = _predict_ml_models(df, symbol=symbol, force_retrain=force_retrain)
    ml_signals = [r["signal_next_day"] for r in ml_results]

    # Votación por mayoría SOLO con modelos ML
    if len(ml_signals) == 0:
        voted = 0
    else:
        count_buy = ml_signals.count(1)
        count_sell = ml_signals.count(-1)
        
        if count_buy > count_sell:
            voted = 1
        elif count_sell > count_buy:
            voted = -1
        else:
            voted = 0  # Empate = neutral

    logger.info(
        f"Ensemble para {symbol} en {df.index[-1].date()}: "
        f"Reglas (info)={rule_signals}, ML signals (votan)={ml_signals}, Final={voted}"
    )

    return {
        "rule_signals": rule_signals,  # Solo informativo
        "ml_models": ml_results,
        "signal_ensemble": int(voted),  # Solo basado en ML
        "ml_signals": ml_signals  # Para transparencia
    }


def compute_signals_for_symbol(symbol: str) -> dict:
    """
    Calcula señales para todas las fechas disponibles y las guarda en la tabla 'signals'.
    
    OPTIMIZADO: Solo usa reglas rápidas para todas las fechas históricas.
    Los modelos ML solo se usan cuando llamas a predict_ensemble.
    
    - signal_simple: basado en la primera regla
    - signal_ensemble: votación de las 3 reglas (rápido)
    
    Devuelve la señal de la última fecha.
    """
    df = _load_features(symbol)
    if df.empty:
        return {
            "symbol": symbol,
            "signal_simple": 0,
            "signal_ensemble": 0,
        }

    conn = get_db_conn()
    last_simple = 0
    last_ensemble = 0
    
    logger.info(f"Calculando señales para {len(df)} fechas de {symbol}...")

    with conn, conn.cursor() as cur:
        for idx, (date, row) in enumerate(df.iterrows()):
            # Calcular las 3 señales basadas en reglas (muy rápido)
            s1 = _rule_based_signal(row)
            s2 = _rule_based_signal_alt(row)
            s3 = _rule_based_signal_contrarian(row)
            
            # Votación simple de las 3 reglas
            signals = np.array([s1, s2, s3], dtype=int)
            mean = signals.mean()
            
            if mean > 0.2:
                voted = 1
            elif mean < -0.2:
                voted = -1
            else:
                voted = 0

            # Guardar en BD
            cur.execute(
                """
                INSERT INTO signals (symbol, date, signal_simple, signal_ensemble, model_best)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE
                  SET signal_simple = EXCLUDED.signal_simple,
                      signal_ensemble = EXCLUDED.signal_ensemble,
                      model_best = EXCLUDED.model_best;
                """,
                (
                    symbol,
                    date.date(),
                    int(s1),
                    int(voted),
                    "rules_ensemble",
                ),
            )

            last_simple = int(s1)
            last_ensemble = int(voted)
            
            # Log cada 100 filas para seguimiento
            if (idx + 1) % 100 == 0:
                logger.info(f"Procesadas {idx + 1}/{len(df)} fechas...")

    logger.info(
        f"✅ Señales calculadas para {symbol}: última fecha {df.index[-1].date()}, "
        f"simple={last_simple}, ensemble={last_ensemble}"
    )

    return {
        "symbol": symbol,
        "signal_simple": last_simple,
        "signal_ensemble": last_ensemble,
    }
