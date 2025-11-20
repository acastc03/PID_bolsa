import feedparser
from datetime import datetime
from .config import get_db_conn
from . import logger

def fetch_google_news(symbol: str, max_items: int = 10):
    query = "IBEX 35" if symbol == "^IBEX" else symbol
    url = f"https://news.google.com/rss/search?q={query}+when:1d&hl=es&gl=ES&ceid=ES:es"
    feed = feedparser.parse(url)
    return feed.entries[:max_items]

def update_news_for_symbols(symbols):
    conn = get_db_conn()
    inserted = 0
    with conn, conn.cursor() as cur:
        for symbol in symbols:
            entries = fetch_google_news(symbol)
            for e in entries:
                published = datetime(*e.published_parsed[:6])
                cur.execute("""
                    INSERT INTO news(symbol, published_at, title, source, url)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (url) DO NOTHING;
                """, (symbol, published, e.title, e.get("source", {}).get("title", ""), e.link))
                inserted += cur.rowcount
    logger.info(f"{inserted} noticias insertadas.")
    return inserted