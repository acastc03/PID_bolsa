# mcp_server/scripts/models.py

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import optuna
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import warnings
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

from .config import get_db_conn
from . import logger
from .model_storage import save_model, load_model, model_exists
from psycopg2 import Error as PsycopgError


def _load_features(symbol: str, as_of_date=None) -> pd.DataFrame:
    """
    Carga precios + indicadores para un símbolo y construye un DataFrame
    de features indexado por fecha.
    
    Args:
        symbol: Símbolo del activo
        as_of_date: Si se especifica (date), solo carga datos hasta esa fecha.
                   Útil para backfill sin look-ahead bias.
    """
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            if as_of_date:
                # Filtrar datos hasta as_of_date (sin información del futuro)
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
                    WHERE p.symbol = %s AND p.date <= %s
                    ORDER BY p.date
                    """,
                    (symbol, as_of_date),
                )
            else:
                # Comportamiento original: todos los datos
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
        # solo lectura, no hace falta commit

    except PsycopgError as e:
        logger.error(f"Error de Postgres al cargar features para {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise
    finally:
        if conn is not None and not conn.closed:
            conn.close()

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
# HYPERPARAMETER TUNING
# ========================================================================

# Espacios de búsqueda de hiperparámetros para cada modelo
PARAM_SPACES = {
    "RandomForest": {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 15),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 5),
        "max_features": Categorical(['sqrt', 'log2', None])
    },
    "XGBoost": {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 10),
        "learning_rate": Real(0.01, 0.3, prior='log-uniform'),
        "subsample": Real(0.6, 1.0),
        "colsample_bytree": Real(0.6, 1.0),
        "gamma": Real(0, 5)
    },
    "SVR": {
        "C": Real(0.1, 100, prior='log-uniform'),
        "gamma": Real(0.001, 1, prior='log-uniform'),
        "epsilon": Real(0.01, 1.0)
    },
    "LightGBM": {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 15),
        "learning_rate": Real(0.01, 0.3, prior='log-uniform'),
        "num_leaves": Integer(10, 100),
        "subsample": Real(0.6, 1.0),
        "colsample_bytree": Real(0.6, 1.0)
    },
    "CatBoost": {
        "iterations": Integer(50, 300),
        "depth": Integer(3, 10),
        "learning_rate": Real(0.01, 0.3, prior='log-uniform'),
        "l2_leaf_reg": Real(1, 10)
    }
}


def optimize_hyperparameters(model_class, param_space, X_train, y_train, model_name: str, n_iter: int = 20):
    """
    Optimiza hiperparámetros usando Bayesian Optimization con BayesSearchCV.
    
    Args:
        model_class: Clase del modelo (ej: RandomForestRegressor)
        param_space: Diccionario con el espacio de búsqueda
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        model_name: Nombre del modelo para logging
        n_iter: Número de iteraciones de búsqueda
    
    Returns:
        Mejores parámetros encontrados
    """
    logger.info(f"🔍 Optimizando hiperparámetros para {model_name}...")
    
    try:
        # Configurar BayesSearchCV
        opt = BayesSearchCV(
            estimator=model_class(),
            search_spaces=param_space,
            n_iter=n_iter,
            cv=3,  # 3-fold cross-validation
            n_jobs=-1,
            scoring='neg_mean_absolute_error',
            random_state=42,
            verbose=0
        )
        
        # Ejecutar búsqueda
        opt.fit(X_train, y_train)
        
        best_params = opt.best_params_
        best_score = -opt.best_score_  # Negativo porque es neg_mean_absolute_error
        
        logger.info(f"✅ {model_name} - Mejor MAE en CV: {best_score:.2f}")
        logger.info(f"📊 Mejores parámetros: {best_params}")
        
        return best_params
        
    except Exception as e:
        logger.error(f"❌ Error optimizando {model_name}: {e}")
        return None


def load_best_params(symbol: str, model_name: str):
    """
    Carga los mejores parámetros guardados para un modelo y símbolo.
    Si no existen, devuelve None.
    """
    try:
        model_data = load_model(symbol, model_name)
        if model_data and "best_params" in model_data.get("metadata", {}):
            return model_data["metadata"]["best_params"]
    except:
        pass
    return None


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

def _predict_ml_models(df: pd.DataFrame, symbol: str = "^IBEX", force_retrain: bool = False, tune_hyperparams: bool = False) -> list:
    """
    Entrena y predice con 7 modelos de ML.
    - Si force_retrain=False, intenta cargar modelos guardados
    - Si no existen, entrena nuevos y los guarda automáticamente
    - Si tune_hyperparams=True, optimiza hiperparámetros antes de entrenar
    
    Args:
        df: DataFrame con features
        symbol: Símbolo del activo
        force_retrain: Si True, fuerza reentrenamiento aunque existan modelos
        tune_hyperparams: Si True, ejecuta hyperparameter tuning (Bayesian optimization)
    
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
                    "training_date": model_data.get("training_date"),
                    "tuned": "best_params" in model_data.get("metadata", {})
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            
            # Optimizar hiperparámetros si está activado
            if tune_hyperparams and model_name in PARAM_SPACES:
                best_params = optimize_hyperparameters(
                    RandomForestRegressor, 
                    PARAM_SPACES[model_name],
                    X_train, y_train, 
                    model_name
                )
            else:
                # Intentar cargar parámetros guardados o usar defaults
                best_params = load_best_params(symbol, model_name)
                if not best_params:
                    best_params = {"n_estimators": 100, "random_state": 42}
            
            # Entrenar con mejores parámetros
            rf = RandomForestRegressor(**best_params, random_state=42).fit(X_train, y_train)
            pred = rf.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, rf.predict(X_train))
            
            metadata = {
                "MAE": float(mae), 
                "RMSE": float(rmse), 
                "n_samples": len(X_train),
                "best_params": best_params
            }
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
                    "training_date": model_data.get("training_date"),
                    "tuned": "best_params" in model_data.get("metadata", {})
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            
            if tune_hyperparams and model_name in PARAM_SPACES:
                best_params = optimize_hyperparameters(
                    XGBRegressor, 
                    PARAM_SPACES[model_name],
                    X_train, y_train, 
                    model_name
                )
            else:
                best_params = load_best_params(symbol, model_name)
                if not best_params:
                    best_params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4}
            
            xgb = XGBRegressor(**best_params, random_state=42)
            xgb.fit(X_train, y_train)
            pred = xgb.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, xgb.predict(X_train))
            
            metadata = {
                "MAE": float(mae), 
                "RMSE": float(rmse), 
                "n_samples": len(X_train),
                "best_params": best_params
            }
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
                    "training_date": model_data.get("training_date"),
                    "tuned": "best_params" in model_data.get("metadata", {})
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            
            if tune_hyperparams and model_name in PARAM_SPACES:
                best_params = optimize_hyperparameters(
                    SVR, 
                    PARAM_SPACES[model_name],
                    X_train, y_train, 
                    model_name
                )
            else:
                best_params = load_best_params(symbol, model_name)
                if not best_params:
                    best_params = {"kernel": "rbf", "C": 100, "gamma": 0.1, "epsilon": 0.1}
            
            svr = SVR(**best_params)
            svr.fit(X_train, y_train)
            pred = svr.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, svr.predict(X_train))
            
            metadata = {
                "MAE": float(mae), 
                "RMSE": float(rmse), 
                "n_samples": len(X_train),
                "best_params": best_params
            }
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
                    "training_date": model_data.get("training_date"),
                    "tuned": "best_params" in model_data.get("metadata", {})
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            
            if tune_hyperparams and model_name in PARAM_SPACES:
                best_params = optimize_hyperparameters(
                    LGBMRegressor, 
                    PARAM_SPACES[model_name],
                    X_train, y_train, 
                    model_name
                )
            else:
                best_params = load_best_params(symbol, model_name)
                if not best_params:
                    best_params = {
                        "n_estimators": 300,
                        "learning_rate": 0.05,
                        "max_depth": -1,
                        "num_leaves": 31
                    }
            
            lgbm = LGBMRegressor(**best_params, random_state=42, verbose=-1)
            lgbm.fit(X_train, y_train)
            pred = lgbm.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, lgbm.predict(X_train))
            
            metadata = {
                "MAE": float(mae), 
                "RMSE": float(rmse), 
                "n_samples": len(X_train),
                "best_params": best_params
            }
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
                    "training_date": model_data.get("training_date"),
                    "tuned": "best_params" in model_data.get("metadata", {})
                })
        else:
            logger.info(f"🔄 Entrenando nuevo modelo: {model_name}")
            
            if tune_hyperparams and model_name in PARAM_SPACES:
                best_params = optimize_hyperparameters(
                    CatBoostRegressor, 
                    PARAM_SPACES[model_name],
                    X_train, y_train, 
                    model_name
                )
            else:
                best_params = load_best_params(symbol, model_name)
                if not best_params:
                    best_params = {
                        "iterations": 300,
                        "learning_rate": 0.05,
                        "depth": 6
                    }
            
            cat = CatBoostRegressor(**best_params, silent=True, random_state=42)
            cat.fit(X_train, y_train)
            pred = cat.predict(X_test)[0]
            mae, rmse = evaluate_model(y_train, cat.predict(X_train))
            
            metadata = {
                "MAE": float(mae), 
                "RMSE": float(rmse), 
                "n_samples": len(X_train),
                "best_params": best_params
            }
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


def predict_ensemble(symbol: str, force_retrain: bool = False, as_of_date=None, tune_hyperparams: bool = False) -> dict:
    """
    Calcula señales con:
    - 3 reglas basadas en indicadores (solo como referencia)
    - 7 modelos de ML (estos son los que votan)
    
    El ensemble se calcula SOLO con los modelos ML.
    
    Args:
        symbol: Símbolo del activo
        force_retrain: Si True, fuerza reentrenamiento de todos los modelos
        as_of_date: Si se especifica (date), solo usa datos hasta esa fecha.
                   Para backfill histórico sin look-ahead bias.
        tune_hyperparams: Si True, optimiza hiperparámetros con Bayesian optimization
    
    Devuelve:
        - rule_signals: señales de las 3 reglas (informativo)
        - ml_models: lista de resultados de los 7 modelos ML
        - signal_ensemble: señal final por votación (SOLO modelos ML)
    """
    df = _load_features(symbol, as_of_date=as_of_date)
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
    # Si as_of_date está presente, SIEMPRE forzar reentrenamiento (no usar modelos guardados)
    force_retrain_internal = force_retrain or (as_of_date is not None)
    ml_results = _predict_ml_models(df, symbol=symbol, force_retrain=force_retrain_internal, tune_hyperparams=tune_hyperparams)
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

    mode = "BACKFILL" if as_of_date else "LIVE"
    logger.info(
        f"[{mode}] Ensemble para {symbol} en {df.index[-1].date()}: "
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

    conn = None
    last_simple = 0
    last_ensemble = 0
    logger.info(f"Calculando señales para {len(df)} fechas de {symbol}...")

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for idx, (date, row) in enumerate(df.iterrows()):
                s1 = _rule_based_signal(row)
                s2 = _rule_based_signal_alt(row)
                s3 = _rule_based_signal_contrarian(row)

                signals = np.array([s1, s2, s3], dtype=int)
                mean = signals.mean()
                if mean > 0.2:
                    voted = 1
                elif mean < -0.2:
                    voted = -1
                else:
                    voted = 0

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

                if (idx + 1) % 100 == 0:
                    logger.info(f"Procesadas {idx + 1}/{len(df)} fechas...")

        conn.commit()
        logger.info(
            f"✅ Señales calculadas para {symbol}: última fecha {df.index[-1].date()}, "
            f"simple={last_simple}, ensemble={last_ensemble}"
        )
    except PsycopgError as e:
        logger.error(f"Error de Postgres al guardar señales de {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise
    finally:
        if conn is not None and not conn.closed:
            conn.close()

    return {
        "symbol": symbol,
        "signal_simple": last_simple,
        "signal_ensemble": last_ensemble,
    }