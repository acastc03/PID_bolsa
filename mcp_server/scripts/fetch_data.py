"""Módulo de descarga de datos financieros desde Yahoo Finance.

Gestiona la obtención de precios históricos (OHLCV) desde yfinance
y su almacenamiento en la base de datos PostgreSQL.
"""

import yfinance as yf
import pandas as pd
from psycopg2 import Error as PsycopgError
from .config import get_db_conn
from . import logger


def _find_col(df: pd.DataFrame, target: str):
    """Busca una columna en un DataFrame de manera flexible.
    
    Intenta encontrar una columna que coincida con 'target' de dos formas:
    1. Coincidencia exacta (ignorando mayúsculas/minúsculas)
    2. Contenga el texto target como substring
    
    Esto es útil porque yfinance puede devolver columnas con diferentes
    formatos dependiendo del contexto (ej: "Close" vs "Close IBEX").
    
    Args:
        df: DataFrame donde buscar
        target: Nombre de columna a buscar (ej: "Close", "Volume")
        
    Returns:
        str | None: Nombre exacto de la columna encontrada, o None si no existe
    """
    cols = [str(c) for c in df.columns]

    # primero coincidencia exacta (case-insensitive)
    for c in cols:
        if c.lower() == target.lower():
            return c

    # luego coincidencia parcial
    for c in cols:
        if target.lower() in c.lower():
            return c

    return None


def update_prices_for_symbol(symbol: str, period: str = "1mo") -> int:
    """Descarga precios históricos desde Yahoo Finance y los guarda en BD.
    
    Obtiene datos OHLCV (Open, High, Low, Close, Volume) para un símbolo
    y los inserta/actualiza en la tabla 'prices' usando UPSERT (ON CONFLICT).
    
    Args:
        symbol: Símbolo de Yahoo Finance (ej: "^IBEX", "^GSPC")
        period: Período de tiempo a descargar. Opciones:
               - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"
               
    Returns:
        int: Número de filas insertadas/actualizadas en la base de datos
        
    Raises:
        RuntimeError: Si faltan columnas esenciales en los datos descargados
        PsycopgError: Si hay error al insertar en PostgreSQL
        
    Note:
        - Usa ON CONFLICT para actualizar filas existentes
        - Maneja columnas MultiIndex automáticamente
        - Convierte NaN en volumen a 0
    """
    logger.info(f"Descargando precios de {symbol} ({period})...")
    df = yf.download(symbol, period=period)

    # Aplanar columnas si vienen como MultiIndex (ocurre con múltiples símbolos)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(c) for c in col if c]).strip()
            for col in df.columns.values
        ]

    logger.info(f"Columnas obtenidas de yfinance para {symbol}: {list(df.columns)}")

    # Verificar que se obtuvieron datos
    if df.empty:
        logger.warning(f"No se han obtenido datos para {symbol}")
        return 0

    # Detectar columnas reales
    open_col = _find_col(df, "Open")
    high_col = _find_col(df, "High")
    low_col = _find_col(df, "Low")
    close_col = _find_col(df, "Close") or _find_col(df, "Adj Close")
    vol_col = _find_col(df, "Volume")

    missing = []
    if not open_col:
        missing.append("Open")
    if not high_col:
        missing.append("High")
    if not low_col:
        missing.append("Low")
    if not close_col:
        missing.append("Close/Adj Close")
    if not vol_col:
        missing.append("Volume")

    if missing:
        raise RuntimeError(
            f"Columnas requeridas no encontradas en datos de {symbol}: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    try:
        conn = get_db_conn()
        with conn, conn.cursor() as cur:
            for date, row in df.iterrows():
                # close/adj_close
                adj_close = float(row[close_col])

                # volume: si es NaN, lo ponemos a 0
                vol_val = row[vol_col]
                if pd.isna(vol_val):
                    volume = 0
                else:
                    volume = int(vol_val)

                cur.execute(
                    """
                    INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (symbol, date) DO UPDATE
                    SET open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        adj_close = EXCLUDED.adj_close,
                        volume = EXCLUDED.volume;
                    """,
                    (
                        symbol,
                        date.date(),
                        float(row[open_col]),
                        float(row[high_col]),
                        float(row[low_col]),
                        float(row[close_col]),
                        adj_close,
                        volume,
                    )
                )
        logger.info(f"Insertadas/actualizadas {len(df)} filas de {symbol}")
        return len(df)

    except PsycopgError as e:
        logger.error(f"Error de Postgres al actualizar precios: {e}")
        raise