#!/usr/bin/env python3
"""Servidor MCP para integración con Claude Desktop.

Este servidor permite que Claude Desktop acceda a las funcionalidades
del sistema de predicción de mercados financieros a través del
Model Context Protocol.

Herramientas disponibles para Claude:
- get_market_price: Obtener último precio de un mercado
- get_prediction: Obtener predicción ML para un mercado
- get_indicators: Obtener indicadores técnicos
- get_news: Obtener últimas noticias
- update_data: Actualizar datos del mercado
- get_daily_summary: Resumen completo del día
"""

import asyncio
import os
import sys
from datetime import date, datetime
from typing import Any, Optional

# Añadir el directorio padre al path para importar scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp import types

# Importar funciones del sistema de predicción
try:
    from mcp_server.scripts.config import get_db_conn
    from mcp_server.scripts.assets import resolve_symbol
    from mcp_server.scripts.fetch_data import update_prices_for_symbol
    from mcp_server.scripts.indicators import compute_indicators_for_symbol
    from mcp_server.scripts.news import update_news_for_symbols
    from mcp_server.scripts.models import predict_ensemble, predict_simple
    from mcp_server.scripts.reporting import build_daily_summary
    from mcp_server.scripts.validate_predictions import validate_predictions_yesterday
except ImportError as e:
    print(f"Error importando módulos: {e}", file=sys.stderr)
    print("Asegúrate de que PYTHONPATH incluye el directorio del proyecto", file=sys.stderr)
    sys.exit(1)


# Inicializar servidor MCP
server = Server("finance-predictor")

# Lista de mercados soportados - sincronizado con Market enum
SUPPORTED_MARKETS = [
    "IBEX35",  # España - IBEX 35
    "SP500",   # USA - S&P 500
    "NIKKEI"   # Japón - Nikkei 225
]


def get_latest_price(symbol: str) -> dict[str, Any]:
    """Obtiene el último precio disponible para un símbolo."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, close, open, high, low, volume
                FROM prices
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1
            """, (symbol,))
            row = cur.fetchone()
            
            if not row:
                return {"error": f"No hay datos de precios para {symbol}"}
            
            return {
                "symbol": symbol,
                "date": row["date"].isoformat(),
                "close": float(row["close"]),
                "open": float(row["open"]) if row["open"] else None,
                "high": float(row["high"]) if row["high"] else None,
                "low": float(row["low"]) if row["low"] else None,
                "volume": int(row["volume"]) if row["volume"] else 0,
            }
    finally:
        conn.close()


def get_latest_indicators(symbol: str) -> dict[str, Any]:
    """Obtiene los últimos indicadores técnicos."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, sma_20, sma_50, vol_20, rsi_14
                FROM indicators
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1
            """, (symbol,))
            row = cur.fetchone()
            
            if not row:
                return {"error": f"No hay indicadores calculados para {symbol}"}
            
            return {
                "symbol": symbol,
                "date": row["date"].isoformat(),
                "sma_20": float(row["sma_20"]) if row["sma_20"] else None,
                "sma_50": float(row["sma_50"]) if row["sma_50"] else None,
                "volatility_20": float(row["vol_20"]) if row["vol_20"] else None,
                "rsi_14": float(row["rsi_14"]) if row["rsi_14"] else None,
            }
    finally:
        conn.close()


def get_recent_news(symbol: str, limit: int = 5) -> list[dict[str, Any]]:
    """Obtiene las últimas noticias."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT published_at, title, source, url, sentiment
                FROM news
                WHERE symbol = %s
                ORDER BY published_at DESC
                LIMIT %s
            """, (symbol, limit))
            rows = cur.fetchall()
            
            return [{
                "published_at": row["published_at"].isoformat() if isinstance(row["published_at"], datetime) else str(row["published_at"]),
                "title": row["title"],
                "source": row["source"],
                "url": row["url"],
                "sentiment": float(row["sentiment"]) if row["sentiment"] else None,
            } for row in rows]
    finally:
        conn.close()


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Lista todas las herramientas disponibles para Claude."""
    return [
        types.Tool(
            name="get_market_price",
            description="""Obtiene el último precio disponible de un mercado financiero.
            
            Mercados soportados (30+ índices globales):
            
            🇪🇺 Europa: IBEX35 (España), FTSE100 (UK), DAX (Alemania), CAC40 (Francia), 
                        FTSEMIB (Italia), EUROSTOXX50
            
            🇺🇸 América: SP500, DOW, NASDAQ, NASDAQ100, RUSSELL2000, VIX (volatilidad),
                         BOVESPA (Brasil), IPC (México)
            
            🌏 Asia-Pacífico: NIKKEI (Japón), HANGSENG (Hong Kong), SHANGHAI (China),
                              SENSEX (India), NIFTY50 (India), ASX200 (Australia), KOSPI (Corea)
            
            Devuelve: fecha, precio de cierre, apertura, máximo, mínimo y volumen.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "description": "Nombre del mercado (ver lista completa en descripción)",
                        "enum": SUPPORTED_MARKETS
                    }
                },
                "required": ["market"]
            }
        ),
        types.Tool(
            name="get_prediction",
            description="""Obtiene la predicción ML más reciente para un mercado.
            
            Utiliza un ensemble de 7 modelos de Machine Learning:
            - LinearRegression
            - RandomForest
            - Prophet
            - XGBoost
            - SVR
            - LightGBM
            - CatBoost
            
            La señal final se decide por votación mayoritaria.
            Señales: +1 (compra), 0 (neutral), -1 (venta)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "description": "Nombre del mercado (30+ índices disponibles)",
                        "enum": SUPPORTED_MARKETS
                    },
                    "force_retrain": {
                        "type": "boolean",
                        "description": "Forzar reentrenamiento de modelos (por defecto: false)",
                        "default": False
                    }
                },
                "required": ["market"]
            }
        ),
        types.Tool(
            name="get_indicators",
            description="""Obtiene los indicadores técnicos más recientes.
            
            Indicadores disponibles:
            - SMA 20 y 50: Medias móviles simples
            - RSI 14: Índice de fuerza relativa (0-100)
            - Volatilidad 20 días: Desviación estándar de retornos
            
            Útil para análisis técnico y toma de decisiones.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "description": "Nombre del mercado (30+ índices disponibles)",
                        "enum": SUPPORTED_MARKETS
                    }
                },
                "required": ["market"]
            }
        ),
        types.Tool(
            name="get_news",
            description="""Obtiene las últimas noticias financieras para un mercado.
            
            Fuentes: Google News RSS y Yahoo Finance.
            Incluye: título, fecha de publicación, fuente, URL y sentiment (si disponible).""",
            inputSchema={
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "description": "Nombre del mercado (30+ índices disponibles)",
                        "enum": SUPPORTED_MARKETS
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Número máximo de noticias (por defecto: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["market"]
            }
        ),
        types.Tool(
            name="update_market_data",
            description="""Actualiza los datos de un mercado desde Yahoo Finance.
            
            Descarga precios históricos recientes y calcula indicadores técnicos.
            Útil para tener los datos más actualizados antes de hacer predicciones.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "description": "Nombre del mercado (30+ índices disponibles)",
                        "enum": SUPPORTED_MARKETS
                    },
                    "period": {
                        "type": "string",
                        "description": "Período a descargar (1d, 5d, 1mo, 3mo, 6mo, 1y)",
                        "default": "5d",
                        "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y"]
                    }
                },
                "required": ["market"]
            }
        ),
        types.Tool(
            name="get_daily_summary",
            description="""Obtiene un resumen completo del día para un mercado.
            
            Incluye:
            - Precio actual y variación
            - Indicadores técnicos
            - Señales de trading
            - Últimas noticias
            - Texto formateado para email/reporte
            
            Ideal para reportes diarios automatizados.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "description": "Nombre del mercado (30+ índices disponibles)",
                        "enum": SUPPORTED_MARKETS
                    }
                },
                "required": ["market"]
            }
        ),
        types.Tool(
            name="validate_predictions",
            description="""Valida las predicciones del día anterior contra valores reales.
            
            Calcula el error absoluto entre predicciones y precios reales.
            Útil para evaluar la precisión de los modelos.""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Maneja las llamadas a herramientas desde Claude."""
    
    if not arguments:
        arguments = {}
    
    try:
        if name == "get_market_price":
            market = arguments["market"]
            symbol = resolve_symbol(market)
            result = get_latest_price(symbol)
            
            return [types.TextContent(
                type="text",
                text=f"📊 Último precio de {market}:\n\n" + 
                     f"• Fecha: {result.get('date', 'N/A')}\n" +
                     f"• Cierre: {result.get('close', 'N/A'):,.2f}\n" +
                     f"• Apertura: {result.get('open', 'N/A'):,.2f}\n" +
                     f"• Máximo: {result.get('high', 'N/A'):,.2f}\n" +
                     f"• Mínimo: {result.get('low', 'N/A'):,.2f}\n" +
                     f"• Volumen: {result.get('volume', 0):,}"
            )]
        
        elif name == "get_prediction":
            market = arguments["market"]
            force_retrain = arguments.get("force_retrain", False)
            symbol = resolve_symbol(market)
            
            result = predict_ensemble(symbol, force_retrain=force_retrain)
            
            # Formatear resultados de modelos
            ml_summary = "\n".join([
                f"  • {m['model_name']}: {m['prediction_next_day']:,.2f} → "
                f"{'🟢 COMPRA' if m['signal_next_day'] == 1 else '🔴 VENTA' if m['signal_next_day'] == -1 else '⚪ NEUTRAL'}"
                for m in result.get("ml_models", [])
            ])
            
            signal_ensemble = result.get("signal_ensemble", 0)
            signal_text = "🟢 COMPRA (+1)" if signal_ensemble == 1 else "🔴 VENTA (-1)" if signal_ensemble == -1 else "⚪ NEUTRAL (0)"
            
            return [types.TextContent(
                type="text",
                text=f"🤖 Predicción ML para {market}:\n\n" +
                     f"📊 Señal del Ensemble: {signal_text}\n\n" +
                     f"Predicciones individuales:\n{ml_summary}\n\n" +
                     f"{'⚡ Modelos reentrenados' if force_retrain else '📦 Usando modelos guardados'}"
            )]
        
        elif name == "get_indicators":
            market = arguments["market"]
            symbol = resolve_symbol(market)
            result = get_latest_indicators(symbol)
            
            if "error" in result:
                return [types.TextContent(type="text", text=f"❌ {result['error']}")]
            
            rsi = result.get("rsi_14")
            rsi_signal = "📈 Sobreventa" if rsi and rsi < 30 else "📉 Sobrecompra" if rsi and rsi > 70 else "➖ Neutral"
            
            return [types.TextContent(
                type="text",
                text=f"📈 Indicadores técnicos de {market}:\n\n" +
                     f"• Fecha: {result.get('date', 'N/A')}\n" +
                     f"• SMA 20: {result.get('sma_20', 'N/A'):,.2f}\n" +
                     f"• SMA 50: {result.get('sma_50', 'N/A'):,.2f}\n" +
                     f"• RSI 14: {result.get('rsi_14', 'N/A'):.1f} {rsi_signal}\n" +
                     f"• Volatilidad 20d: {result.get('volatility_20', 'N/A'):.4f}"
            )]
        
        elif name == "get_news":
            market = arguments["market"]
            limit = arguments.get("limit", 5)
            symbol = resolve_symbol(market)
            news_list = get_recent_news(symbol, limit)
            
            if not news_list:
                return [types.TextContent(
                    type="text",
                    text=f"📰 No hay noticias recientes para {market}"
                )]
            
            news_text = "\n\n".join([
                f"{i+1}. {n['title']}\n   📅 {n['published_at']}\n   🔗 {n['url']}"
                for i, n in enumerate(news_list)
            ])
            
            return [types.TextContent(
                type="text",
                text=f"📰 Últimas {len(news_list)} noticias de {market}:\n\n{news_text}"
            )]
        
        elif name == "update_market_data":
            market = arguments["market"]
            period = arguments.get("period", "5d")
            symbol = resolve_symbol(market)
            
            # Actualizar precios
            rows_prices = update_prices_for_symbol(symbol, period)
            
            # Calcular indicadores
            rows_indicators = compute_indicators_for_symbol(symbol)
            
            return [types.TextContent(
                type="text",
                text=f"✅ Datos actualizados para {market}:\n\n" +
                     f"• Precios: {rows_prices} filas actualizadas\n" +
                     f"• Indicadores: {rows_indicators} filas calculadas\n" +
                     f"• Período: {period}"
            )]
        
        elif name == "get_daily_summary":
            market = arguments["market"]
            symbol = resolve_symbol(market)
            summary = build_daily_summary(symbol)
            
            return [types.TextContent(
                type="text",
                text=summary.get("email_text", "No hay resumen disponible")
            )]
        
        elif name == "validate_predictions":
            result = validate_predictions_yesterday()
            
            return [types.TextContent(
                type="text",
                text=f"✅ Predicciones validadas:\n\n" +
                     f"• Fecha objetivo: {result['target_date']}\n" +
                     f"• Símbolos con precio: {', '.join(result['symbols_with_price'])}\n" +
                     f"• Filas actualizadas: {result['rows_updated']}"
            )]
        
        else:
            return [types.TextContent(
                type="text",
                text=f"❌ Herramienta desconocida: {name}"
            )]
    
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Error ejecutando {name}: {str(e)}"
        )]


async def main():
    """Punto de entrada principal del servidor MCP."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="finance-predictor",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
