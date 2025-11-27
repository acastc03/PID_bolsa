"""Módulo de limpieza y preprocesamiento de datos de precios.

Proporciona funciones para normalizar y limpiar series temporales
de precios financieros antes de análisis o modelado.
"""

import pandas as pd

def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y normaliza un DataFrame de precios.
    
    Operaciones realizadas:
    1. Ordena por índice (fecha)
    2. Elimina filas sin precio de cierre
    3. Rellena días hábiles faltantes (festivos, fines de semana)
    4. Forward fill para precios en días sin trading
    5. Calcula retornos diarios
    
    Args:
        df: DataFrame con columna 'Close' indexado por fecha
        
    Returns:
        pd.DataFrame: DataFrame limpio con frecuencia de días hábiles
                     y columna adicional 'return_1d' (retornos diarios)
                     
    Note:
        - asfreq("B"): Business days (días hábiles Mon-Fri)
        - ffill(): Forward fill para mantener último precio conocido
        - Útil antes de entrenar modelos ML
    """
    # Ordenar cronológicamente
    df = df.sort_index()
    
    # Eliminar filas sin datos de cierre
    df = df.dropna(subset=["Close"])
    
    # Establecer frecuencia de días hábiles (rellenando huecos)
    df = df.asfreq("B")  # días hábiles (Mon-Fri)
    
    # Rellenar festivos/fines de semana con último valor conocido
    df["Close"] = df["Close"].ffill()
    
    # Calcular retornos diarios (útil para modelos)
    df["return_1d"] = df["Close"].pct_change()
    
    return df