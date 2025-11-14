import feedparser

def fetch_news(q="IBEX 35 OR Bolsa de Madrid", when="7d"):
    q_enc = q.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={q_enc}+when:{when}&hl=es&gl=ES&ceid=ES:es"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries:
        items.append({
            "title": e.get("title"),
            "link": e.get("link"),
            "published": e.get("published", "")
        })
    return items
