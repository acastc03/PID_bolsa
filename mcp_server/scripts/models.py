import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# ---------- MODELO SIMPLE ----------
def predict_simple(df):
    X = df[["sma20", "sma50"]].values
    y = df["Close"].values
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict([X[-1]])[0]
    signal = 1 if pred > y[-1] else -1
    return {"prediction_next_day": float(pred), "signal_next_day": signal}

# ---------- ENSEMBLE COMPLETO ----------
def predict_ensemble(df):
    results = []
    X = df[["sma20", "sma50", "ema10", "ema50", "momentum", "volatility"]]
    y = df["Close"]

    X_train, X_test = X[:-1], X[-1:]
    y_train, y_test = y[:-1], y[-1:]

    # 1️⃣ Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(X_test)[0]
    lr_mae, lr_rmse = evaluate_model(y_train, lr.predict(X_train))
    results.append({
        "model_name": "LinearRegression",
        "prediction_next_day": float(lr_pred),
        "signal_next_day": 1 if lr_pred > y_test.iloc[0] else -1,
        "MAE": float(lr_mae),
        "RMSE": float(lr_rmse)
    })

    # 2️⃣ Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_pred = rf.predict(X_test)[0]
    rf_mae, rf_rmse = evaluate_model(y_train, rf.predict(X_train))
    results.append({
        "model_name": "RandomForest",
        "prediction_next_day": float(rf_pred),
        "signal_next_day": 1 if rf_pred > y_test.iloc[0] else -1,
        "MAE": float(rf_mae),
        "RMSE": float(rf_rmse)
    })

    # 3️⃣ Prophet
    prophet_df = pd.DataFrame({"ds": df["Date"], "y": df["Close"]})
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(prophet_df)
    future = prophet.make_future_dataframe(periods=1)
    forecast = prophet.predict(future)
    prophet_pred = forecast["yhat"].iloc[-1]
    prophet_mae, prophet_rmse = evaluate_model(prophet_df["y"], forecast["yhat"][:-1])
    results.append({
        "model_name": "Prophet",
        "prediction_next_day": float(prophet_pred),
        "signal_next_day": 1 if prophet_pred > y_test.iloc[0] else -1,
        "MAE": float(prophet_mae),
        "RMSE": float(prophet_rmse)
    })

    # 4️⃣ XGBoost
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)[0]
    xgb_mae, xgb_rmse = evaluate_model(y_train, xgb.predict(X_train))
    results.append({
        "model_name": "XGBoost",
        "prediction_next_day": float(xgb_pred),
        "signal_next_day": 1 if xgb_pred > y_test.iloc[0] else -1,
        "MAE": float(xgb_mae),
        "RMSE": float(xgb_rmse)
    })

    # 5️⃣ SVR
    svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train, y_train)
    svr_pred = svr.predict(X_test)[0]
    svr_mae, svr_rmse = evaluate_model(y_train, svr.predict(X_train))
    results.append({
        "model_name": "SVR",
        "prediction_next_day": float(svr_pred),
        "signal_next_day": 1 if svr_pred > y_test.iloc[0] else -1,
        "MAE": float(svr_mae),
        "RMSE": float(svr_rmse)
    })

    # 6️⃣ LightGBM
    lgbm = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42
    )
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)[0]
    lgbm_mae, lgbm_rmse = evaluate_model(y_train, lgbm.predict(X_train))
    results.append({
        "model_name": "LightGBM",
        "prediction_next_day": float(lgbm_pred),
        "signal_next_day": 1 if lgbm_pred > y_test.iloc[0] else -1,
        "MAE": float(lgbm_mae),
        "RMSE": float(lgbm_rmse)
    })

    # 7️⃣ CatBoost
    cat = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        silent=True,
        random_state=42
    )
    cat.fit(X_train, y_train)
    cat_pred = cat.predict(X_test)[0]
    cat_mae, cat_rmse = evaluate_model(y_train, cat.predict(X_train))
    results.append({
        "model_name": "CatBoost",
        "prediction_next_day": float(cat_pred),
        "signal_next_day": 1 if cat_pred > y_test.iloc[0] else -1,
        "MAE": float(cat_mae),
        "RMSE": float(cat_rmse)
    })

    # ---------- ENSEMBLE ----------
    signals = [r["signal_next_day"] for r in results]
    majority_signal = 1 if signals.count(1) > signals.count(-1) else -1
    results.append({
        "model_name": "EnsembleMajority",
        "signal_next_day": majority_signal
    })

    return results
