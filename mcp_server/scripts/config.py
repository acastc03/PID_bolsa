"""Módulo de configuración de base de datos.

Gestiona la conexión a PostgreSQL utilizando variables de entorno
y proporciona una conexión cacheada para mejorar el rendimiento.
"""

import os
from functools import lru_cache
import psycopg2
from psycopg2.extras import RealDictCursor

# Variables de entorno para conexión a PostgreSQL
# Valores por defecto apuntan al contenedor Docker
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "indices")
DB_USER = os.getenv("DB_USER", "finanzas")
DB_PASS = os.getenv("DB_PASS", "finanzas_pass")

@lru_cache(maxsize=1)
def get_db_conn():
    """Obtiene una conexión a la base de datos PostgreSQL.
    
    Utiliza @lru_cache para reutilizar la misma conexión y evitar
    crear múltiples conexiones innecesarias. Los resultados se devuelven
    como diccionarios gracias a RealDictCursor.
    
    Returns:
        psycopg2.connection: Conexión activa a PostgreSQL con cursor tipo dict
        
    Note:
        La conexión se cachea, por lo que llamadas sucesivas retornan
        la misma instancia de conexión.
    """
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        cursor_factory=RealDictCursor,  # Devuelve resultados como dict
    )
    return conn