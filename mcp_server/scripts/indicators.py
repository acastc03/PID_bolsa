# mcp_server/scripts/indicators.py

import pandas as pd
from .config import get_db_conn
from . import logger


def _load_prices(symbol: str) -> pd.DataFrame:
    """
    Carga precios de la tabla 'prices' para un símbolo dado,
    ordenados por fecha.
    """
    conn = get_db_conn()
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT date, close
            FROM prices
            WHERE symbol = %s
            ORDER BY date
            """,
            (symbol,),
        )
        rows = cur.fetchall()

    if not rows:
        logger.warning(f"No hay precios en BD para {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.rename(columns={"close": "Close"}, inplace=True)  # para usar la misma convención
    return df


def _compute_indicators_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula SMA, volatilidad y RSI sobre un DataFrame con columna 'Close'.
    Devuelve un DataFrame con columnas: sma_20, sma_50, vol_20, rsi_14.
    """
    out = pd.DataFrame(index=df.index.copy())
    close = df["Close"]

    # Retorno diario
    returns = close.pct_change()

    # Medias móviles
    out["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    out["sma_50"] = close.rolling(window=50, min_periods=50).mean()

    # Volatilidad 20 días (desviación estándar de retornos)
    out["vol_20"] = returns.rolling(window=20, min_periods=20).std()

    # RSI 14 (versión sencilla tipo Wilder)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    roll_up = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()

    rs = roll_up / roll_down
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    return out


def compute_indicators_for_symbol(symbol: str) -> int:
    """
    Carga precios de 'prices', calcula indicadores y los upsertea en la tabla 'indicators'.
    Devuelve cuántas filas se han insertado/actualizado.
    """
    df_prices = _load_prices(symbol)
    if df_prices.empty:
        return 0

    ind_df = _compute_indicators_df(df_prices)

    # Nos quedamos solo con filas donde haya al menos algún indicador no-NaN
    ind_df = ind_df.dropna(how="all")
    if ind_df.empty:
        logger.warning(f"No se han podido calcular indicadores para {symbol} (muy pocos datos)")
        return 0

    conn = get_db_conn()
    with conn, conn.cursor() as cur:
        for date, row in ind_df.iterrows():
            cur.execute(
                """
                INSERT INTO indicators (symbol, date, sma_20, sma_50, vol_20, rsi_14)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE
                SET sma_20 = EXCLUDED.sma_20,
                    sma_50 = EXCLUDED.sma_50,
                    vol_20 = EXCLUDED.vol_20,
                    rsi_14 = EXCLUDED.rsi_14;
                """,
                (
                    symbol,
                    date.date(),
                    float(row["sma_20"]) if pd.notna(row["sma_20"]) else None,
                    float(row["sma_50"]) if pd.notna(row["sma_50"]) else None,
                    float(row["vol_20"]) if pd.notna(row["vol_20"]) else None,
                    float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else None,
                ),
            )

    logger.info(f"Indicadores calculados/actualizados para {symbol}: {len(ind_df)} filas")
    return len(ind_df)