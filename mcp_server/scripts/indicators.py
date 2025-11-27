# mcp_server/scripts/indicators.py
"""Módulo de cálculo de indicadores técnicos.

Calcula indicadores técnicos comunes utilizados en análisis financiero:
- SMA (Simple Moving Averages): Medias móviles simples
- RSI (Relative Strength Index): Índice de fuerza relativa
- Volatilidad: Desviación estándar de retornos

Todos los indicadores se guardan en la tabla 'indicators' de la BD.
"""

import pandas as pd
from psycopg2 import Error as PsycopgError
from .config import get_db_conn
from . import logger


def _load_prices(symbol: str) -> pd.DataFrame:
    """Carga precios de cierre desde la base de datos.
    
    Recupera todos los precios históricos de un símbolo para
    poder calcular indicadores técnicos que requieren ventanas
    de tiempo (ej: SMA de 50 días).
    
    Args:
        symbol: Símbolo del activo (ej: "^IBEX")
        
    Returns:
        pd.DataFrame: DataFrame indexado por fecha con columna 'Close'
                     vacío si no hay datos
    """
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
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
        # solo lectura → no hace falta commit
    except PsycopgError as e:
        logger.error(f"Error de Postgres al cargar precios de {symbol}: {e}")
        # en lectura normalmente no hace falta rollback, pero por si acaso:
        if conn is not None and not conn.closed:
            conn.rollback()
        raise
    finally:
        if conn is not None and not conn.closed:
            conn.close()

    if not rows:
        logger.warning(f"No hay precios en BD para {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.rename(columns={"close": "Close"}, inplace=True)  # para usar la misma convención
    return df


def _compute_indicators_df(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores técnicos sobre una serie de precios.
    
    Implementa los siguientes indicadores:
    - SMA 20 y 50: Medias móviles simples (tendencia)
    - Volatilidad 20: Desv. estándar de retornos (riesgo)
    - RSI 14: Índice de fuerza relativa (momentum)
    
    Args:
        df: DataFrame con columna 'Close' indexado por fecha
        
    Returns:
        pd.DataFrame: DataFrame con columnas [sma_20, sma_50, vol_20, rsi_14]
                     Valores iniciales serán NaN hasta completar ventanas
                     
    Note:
        - SMA requiere min_periods para evitar cálculos con pocos datos
        - RSI implementa método Wilder (media simple de ganancias/pérdidas)
        - Volatilidad basada en desv. estándar de retornos, no precios
    """
    out = pd.DataFrame(index=df.index.copy())
    close = df["Close"]

    returns = close.pct_change()
    out["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    out["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    out["vol_20"] = returns.rolling(window=20, min_periods=20).std()

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Promedio de ganancias y pérdidas
    roll_up = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()
    rs = roll_up / roll_down
    # RSI normalizado 0-100
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    return out


def compute_indicators_for_symbol(symbol: str) -> int:
    """Calcula indicadores técnicos y los guarda en la base de datos.
    
    Flujo completo:
    1. Carga todos los precios históricos del símbolo
    2. Calcula SMA, RSI y volatilidad para cada fecha
    3. Guarda/actualiza resultados en tabla 'indicators' (UPSERT)
    
    Args:
        symbol: Símbolo del activo (ej: "^IBEX")
        
    Returns:
        int: Número de filas insertadas/actualizadas
             0 si no había datos suficientes
             
    Note:
        - Usa ON CONFLICT para actualizar indicadores existentes
        - Elimina filas donde TODOS los indicadores son NaN
        - Requiere al menos ~50 días de datos para SMA50
    """
    df_prices = _load_prices(symbol)
    if df_prices.empty:
        return 0

    ind_df = _compute_indicators_df(df_prices)
    ind_df = ind_df.dropna(how="all")
    if ind_df.empty:
        logger.warning(f"No se han podido calcular indicadores para {symbol} (muy pocos datos)")
        return 0

    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
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
        conn.commit()
        logger.info(f"Indicadores calculados/actualizados para {symbol}: {len(ind_df)} filas")
        return len(ind_df)

    except PsycopgError as e:
        logger.error(f"Error de Postgres al guardar indicadores de {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise

    finally:
        if conn is not None and not conn.closed:
            conn.close()
