# mcp_server/scripts/news.py
"""Módulo de descarga y almacenamiento de noticias financieras.

Integra dos fuentes de noticias:
1. Google News RSS: Búsqueda flexible con queries
2. Yahoo Finance API: Noticias específicas por símbolo

Todas las noticias se guardan en la tabla 'news' con deduplicación por URL.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from psycopg2 import Error as PsycopgError

import yfinance as yf
import feedparser

from .config import get_db_conn
from . import logger


# ------------------------
#  A) Google News (RSS)
# ------------------------

def fetch_news_rss(q: str = "IBEX 35 OR Bolsa de Madrid", when: str = "7d") -> List[Dict[str, Any]]:
    """Descarga noticias desde Google News RSS.
    
    Utiliza el servicio RSS de Google News con búsqueda personalizada.
    Útil para noticias generales del mercado o keywords específicas.
    
    Args:
        q: Query de búsqueda. Soporta operadores OR, AND, comillas
           Ejemplo: "IBEX 35 OR Bolsa de Madrid"
        when: Ventana temporal. Opciones: "1d", "7d", "30d"
        
    Returns:
        List[Dict]: Lista de noticias con campos:
                   - title: Título
                   - link: URL
                   - published: datetime
                   - source: Fuente
                   
    Note:
        Google News RSS no incluye resumen (summary) útil
    """
    q_enc = q.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={q_enc}+when:{when}&hl=es&gl=ES&ceid=ES:es"
    logger.info(f"Descargando RSS de Google News: {url}")
    feed = feedparser.parse(url)

    items: List[Dict[str, Any]] = []
    for e in feed.entries:
        # Intentamos sacar una fecha razonable
        published_dt: Optional[datetime] = None
        if getattr(e, "published_parsed", None) is not None:
            published_dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        elif getattr(e, "updated_parsed", None) is not None:
            published_dt = datetime(*e.updated_parsed[:6], tzinfo=timezone.utc)

        items.append(
            {
                "title": e.get("title"),
                "link": e.get("link"),
                "published": published_dt,
                "raw_published": e.get("published", ""),
                "source": getattr(e, "source", {}).get("title")
                if isinstance(getattr(e, "source", {}), dict)
                else None,
            }
        )

    return items


from psycopg2 import Error as PsycopgError
...
def fetch_and_store_news_rss(
    symbol: str,
    q: Optional[str] = None,
    when: str = "7d",
    max_items: int = 10,
) -> int:

    """Descarga noticias desde Google News RSS y las guarda en BD.
    
    Workflow:
    1. Genera query de búsqueda (o usa la proporcionada)
    2. Descarga noticias vía RSS
    3. Guarda en tabla 'news' con deduplicación por URL
    
    Args:
        symbol: Símbolo a asociar con las noticias (ej: "^IBEX")
        q: Query personalizada. Si None, genera automáticamente
        when: Ventana temporal ("1d", "7d", "30d")
        max_items: Máximo de noticias a guardar
        
    Returns:
        int: Número de noticias insertadas/actualizadas
        
    Note:
        - Usa ON CONFLICT(url) para evitar duplicados
        - Sentiment se deja como None (placeholder para futuro)
    """

    if q is None:
        q = f"{symbol} OR IBEX 35 OR Bolsa de Madrid"

    items = fetch_news_rss(q=q, when=when)
    if not items:
        logger.warning(f"RSS: no se han obtenido noticias para query={q}")
        return 0

    conn = None
    inserted = 0
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for item in items[:max_items]:
                title = item["title"] or "(sin título)"
                url = item["link"]
                source = item["source"]
                published_at = item["published"] or datetime.now(timezone.utc)
                summary = None  # Google News RSS no trae resumen corto útil

                if not url:
                    continue

                cur.execute(
                    """
                    INSERT INTO news (symbol, published_at, title, source, url, summary, sentiment)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE
                    SET symbol = EXCLUDED.symbol,
                        published_at = EXCLUDED.published_at,
                        title = EXCLUDED.title,
                        source = EXCLUDED.source,
                        summary = EXCLUDED.summary;
                    """,
                    (
                        symbol,
                        published_at,
                        title,
                        source,
                        url,
                        summary,
                        None,  # sentiment placeholder
                    ),
                )
                inserted += 1

        conn.commit()
        logger.info(f"RSS: noticias guardadas/actualizadas para {symbol}: {inserted}")
        return inserted

    except PsycopgError as e:
        logger.error(f"Error guardando noticias RSS para {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise

    finally:
        if conn is not None and not conn.closed:
            conn.close()



# ------------------------
#  B) yfinance (opcional)
# ------------------------

def fetch_and_store_news_yf(
    symbol: str,
    days_back: int = 7,
    max_items: int = 10,
) -> int:
    """Descarga noticias desde Yahoo Finance API y las guarda en BD.
    
    Utiliza la API oficial de yfinance para obtener noticias
    específicas del símbolo. Generalmente más relevantes que RSS.
    
    Args:
        symbol: Símbolo de Yahoo Finance (ej: "^IBEX")
        days_back: Número de días hacia atrás para filtrar
        max_items: Límite de noticias a guardar
        
    Returns:
        int: Número de noticias insertadas/actualizadas
        
    Note:
        - Yahoo Finance incluye summary (resumen)
        - Usa timestamp Unix (providerPublishTime)
        - Deduplica por URL automáticamente
    """
    logger.info(f"Descargando noticias (yfinance) para {symbol} (últimos {days_back} días)...")
    ticker = yf.Ticker(symbol)

    try:
        raw_news: List[Dict[str, Any]] = ticker.news or []
    except Exception as e:
        logger.error(f"Error obteniendo noticias de yfinance para {symbol}: {e}")
        return 0

    if not raw_news:
        logger.warning(f"yfinance: no se han obtenido noticias para {symbol}")
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    conn = None
    inserted = 0
    used = 0
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for item in raw_news:
                if used >= max_items:
                    break

                ts = item.get("providerPublishTime")
                if ts is None:
                    continue

                published_at = datetime.fromtimestamp(ts, tz=timezone.utc)
                if published_at < cutoff:
                    continue

                title = item.get("title") or "(sin título)"
                url = item.get("link")
                source = item.get("publisher")
                summary = item.get("summary") or None

                if not url:
                    continue

                cur.execute(
                    """
                    INSERT INTO news (symbol, published_at, title, source, url, summary, sentiment)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE
                    SET symbol = EXCLUDED.symbol,
                        published_at = EXCLUDED.published_at,
                        title = EXCLUDED.title,
                        source = EXCLUDED.source,
                        summary = EXCLUDED.summary;
                    """,
                    (
                        symbol,
                        published_at,
                        title,
                        source,
                        url,
                        summary,
                        None,
                    ),
                )
                inserted += 1
                used += 1

        conn.commit()
        logger.info(f"yfinance: noticias guardadas/actualizadas para {symbol}: {inserted}")
        return inserted

    except PsycopgError as e:
        logger.error(f"Error guardando noticias yfinance para {symbol}: {e}")
        if conn is not None and not conn.closed:
            conn.rollback()
        raise

    finally:
        if conn is not None and not conn.closed:
            conn.close()


#------------------------
#  C) Función combinada
#------------------------

def update_news_for_symbols(
    symbols: list[str],
    when: str = "7d",
    days_back: int = 7,
    max_items_rss: int = 10,
    max_items_yf: int = 10,
):
    """Descarga noticias de múltiples fuentes para varios símbolos.
    
    Estrategia dual:
    - RSS de Google News: Cobertura amplia, contexto general
    - yfinance API: Noticias específicas, mayor relevancia
    
    Args:
        symbols: Lista de símbolos (ej: ["^IBEX", "^GSPC"])
        when: Ventana para RSS ("1d", "7d", "30d")
        days_back: Días hacia atrás para yfinance
        max_items_rss: Límite por símbolo vía RSS
        max_items_yf: Límite por símbolo vía yfinance
        
    Returns:
        dict: {
            "total": int,  # Total de noticias
            "per_symbol": {  # Detalle por símbolo
                "^IBEX": {"rss": 5, "yfinance": 8, "total": 13},
                ...
            }
        }
        
    Note:
        Ambas fuentes se complementan para máxima cobertura
    """
    per_symbol: dict[str, dict] = {}
    total = 0

    for sym in symbols:
        # Query por defecto para RSS si no se pasa q explícita
        q_default = f"{sym} OR IBEX 35 OR Bolsa de Madrid"

        rss_count = fetch_and_store_news_rss(
            symbol=sym,
            q=q_default,
            when=when,
            max_items=max_items_rss,
        )

        yf_count = fetch_and_store_news_yf(
            symbol=sym,
            days_back=days_back,
            max_items=max_items_yf,
        )

        per_symbol[sym] = {
            "rss": rss_count,
            "yfinance": yf_count,
            "total": rss_count + yf_count,
        }
        total += rss_count + yf_count

    return {
        "total": total,
        "per_symbol": per_symbol,
    }