# mcp_server/scripts/fetch_data.py

"""Módulo de descarga de datos financieros desde Yahoo Finance.

Gestiona la obtención de precios históricos (OHLCV) desde yfinance
y su almacenamiento en la base de datos PostgreSQL.
"""
import yfinance as yf
import pandas as pd
from psycopg2 import Error as PsycopgError
import threading
import time
import os
import fcntl

from .config import get_db_conn
from . import logger

# Lock global para serializar descargas de yfinance (dentro del mismo proceso)
_yf_download_lock = threading.Lock()

# File lock para serializar entre procesos (cuando FastAPI usa múltiples workers)
_LOCK_FILE = "/tmp/yfinance_download.lock"


def _download_with_lock(symbol: str, period: str) -> pd.DataFrame:
    """Descarga datos de yfinance con lock de archivo para evitar race conditions.
    
    Usa tanto threading.Lock (mismo proceso) como file lock (entre procesos)
    para garantizar que solo una descarga de yfinance ocurra a la vez.
    """
    with _yf_download_lock:
        # File lock para sincronizar entre procesos
        lock_file = open(_LOCK_FILE, 'w')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            logger.info(f"Lock adquirido para descargar {symbol}")
            
            # Pequeña pausa para evitar rate limiting de Yahoo Finance
            time.sleep(0.5)
            
            df = yf.download(symbol, period=period, progress=False)
            
            logger.info(f"Descarga completada para {symbol}, filas: {len(df)}")
            return df
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


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
        - Usa lock para evitar race conditions con yfinance en llamadas paralelas
    """
    logger.info(f"Descargando precios de {symbol} ({period})...")
    
    # Lock para evitar que yfinance mezcle datos cuando hay llamadas paralelas
    with _yf_download_lock:
        logger.info(f"Lock adquirido para {symbol}, iniciando descarga...")
        # Pequeña pausa para evitar rate limiting
        time.sleep(0.3)
        df = yf.download(symbol, period=period, progress=False)
        logger.info(f"Descarga finalizada para {symbol}, liberando lock...")

    # Aplanar columnas si vienen como MultiIndex (ocurre con múltiples símbolos)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(c) for c in col if c]).strip()
            for col in df.columns.values
        ]

    logger.info(f"Columnas obtenidas de yfinance para {symbol}: {list(df.columns)}")

    # VALIDACIÓN: Verificar que las columnas corresponden al símbolo correcto
    col_str = " ".join(str(c) for c in df.columns)
    # Si el símbolo solicitado NO está en las columnas pero hay otro símbolo, hay un bug
    other_symbols = ['^GSPC', '^IBEX', '^N225']
    for other in other_symbols:
        if other != symbol and other in col_str and symbol not in col_str:
            logger.error(f"⚠️ BUG DETECTADO: Pedimos {symbol} pero recibimos datos de {other}")
            raise RuntimeError(f"yfinance devolvió datos incorrectos: pedido {symbol}, recibido {other}")

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

    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for date, row in df.iterrows():
                # close / adj_close
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
                    SET open      = EXCLUDED.open,
                        high      = EXCLUDED.high,
                        low       = EXCLUDED.low,
                        close     = EXCLUDED.close,
                        adj_close = EXCLUDED.adj_close,
                        volume    = EXCLUDED.volume;
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
                    ),
                )

        conn.commit()
        logger.info(f"Insertadas/actualizadas {len(df)} filas de {symbol}")
        return len(df)

    except PsycopgError as e:
        logger.error(f"Error de Postgres al actualizar precios: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise

    finally:
        if conn is not None and not conn.closed:
            conn.close()
