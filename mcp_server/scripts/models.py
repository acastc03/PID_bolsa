# mcp_server/scripts/models.py

import numpy as np
import pandas as pd
from .config import get_db_conn
from . import logger


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
             AND p.date   = i.date
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
    return df


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

    # Si faltan datos clave, no damos señal
    if pd.isna(close) or pd.isna(sma20) or pd.isna(rsi):
        return 0

    # Señal alcista
    if (close > sma20) and (40 <= rsi <= 70):
        return 1

    # Señal bajista
    if (close < sma20) and (30 <= rsi <= 60):
        return -1

    # Neutral
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

    # Umbral de volatilidad arbitrario
    if pd.isna(vol20):
        vol20 = 0.01

    # Alcista: por encima de SMA20, baja vol y RSI moderado
    if (close > sma20) and (vol20 < 0.01) and (rsi < 65):
        return 1

    # Bajista: por debajo de SMA20 y alta vol o RSI muy alto (sobrecompra)
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


def predict_simple(symbol: str) -> int:
    """
    Devuelve la señal simple (+1, 0, -1) para la última fecha disponible.
    """
    df = _load_features(symbol)
    if df.empty:
        return 0

    last_row = df.iloc[-1]
    sig = _rule_based_signal(last_row)
    logger.info(f"Señal simple para {symbol} en {df.index[-1].date()}: {sig}")
    return int(sig)


def predict_ensemble(symbol: str) -> dict:
    """
    Calcula 3 señales de reglas distintas y hace una votación por mayoría.
    Devuelve:
      - signals: lista de señales individuales
      - signal_ensemble: señal final (+1, 0, -1)
    """
    df = _load_features(symbol)
    if df.empty:
        return {"signals": [], "signal_ensemble": 0}

    last_row = df.iloc[-1]

    s1 = _rule_based_signal(last_row)
    s2 = _rule_based_signal_alt(last_row)
    s3 = _rule_based_signal_contrarian(last_row)

    signals = np.array([s1, s2, s3], dtype=int)
    if len(signals) == 0:
        voted = 0
    else:
        mean = signals.mean()
        if mean > 0.2:
            voted = 1
        elif mean < -0.2:
            voted = -1
        else:
            voted = 0

    logger.info(
        f"Señales ensemble para {symbol} en {df.index[-1].date()}: "
        f"{signals.tolist()} -> {voted}"
    )

    return {
        "signals": signals.tolist(),
        "signal_ensemble": int(voted),
    }


def compute_signals_for_symbol(symbol: str) -> dict:
    """
    Calcula señales para todas las fechas disponibles y las guarda en la tabla 'signals'.
    Devuelve la señal de la última fecha (simple + ensemble).
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

    with conn, conn.cursor() as cur:
        for date, row in df.iterrows():
            s1 = _rule_based_signal(row)
            s2 = _rule_based_signal_alt(row)
            s3 = _rule_based_signal_contrarian(row)
            signals = np.array([s1, s2, s3], dtype=int)

            if len(signals) == 0:
                voted = 0
            else:
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
                SET signal_simple   = EXCLUDED.signal_simple,
                    signal_ensemble = EXCLUDED.signal_ensemble,
                    model_best      = EXCLUDED.model_best;
                """,
                (
                    symbol,
                    date.date(),
                    int(s1),
                    int(voted),
                    "rules_v1",  # etiqueta del "modelo"
                ),
            )

            last_simple = int(s1)
            last_ensemble = int(voted)

    logger.info(
        f"Señales calculadas para {symbol}: última fecha {df.index[-1].date()}, "
        f"simple={last_simple}, ensemble={last_ensemble}"
    )

    return {
        "symbol": symbol,
        "signal_simple": last_simple,
        "signal_ensemble": last_ensemble,
    }