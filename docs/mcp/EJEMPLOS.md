# 💬 Ejemplos de Conversación con Claude

Una vez configurado el servidor MCP, puedes interactuar con tu sistema de predicción de mercados de forma conversacional con Claude Desktop.

## 📊 Consultar Precios

**Tu pregunta:**
> ¿Cuál es el precio actual del IBEX35?

**Respuesta de Claude:**
```
📊 Último precio de IBEX35:

• Fecha: 2025-12-03
• Cierre: 11,543.21
• Apertura: 11,498.50
• Máximo: 11,567.89
• Mínimo: 11,475.32
• Volumen: 234,567,890
```

---

## 🤖 Obtener Predicción ML

**Tu pregunta:**
> Dame la predicción de Machine Learning para el S&P 500

**Respuesta de Claude:**
```
🤖 Predicción ML para SP500:

📊 Señal del Ensemble: 🟢 COMPRA (+1)

Predicciones individuales:
  • LinearRegression: 5,912.45 → 🟢 COMPRA
  • RandomForest: 5,908.22 → 🔴 VENTA
  • Prophet: 5,915.67 → 🟢 COMPRA
  • XGBoost: 5,910.34 → 🟢 COMPRA
  • SVR: 5,913.89 → 🟢 COMPRA
  • LightGBM: 5,911.56 → 🟢 COMPRA
  • CatBoost: 5,914.12 → 🟢 COMPRA

📦 Usando modelos guardados
```

Claude puede interpretar: "La mayoría de los modelos (6 de 7) sugieren COMPRA, con predicciones de precio alrededor de 5,912 puntos."

---

## 📈 Análisis Técnico

**Tu pregunta:**
> Muéstrame los indicadores técnicos del NASDAQ y explícamelos

**Respuesta de Claude:**
```
📈 Indicadores técnicos de NASDAQ:

• Fecha: 2025-12-03
• SMA 20: 19,234.56
• SMA 50: 18,987.23
• RSI 14: 58.3 ➖ Neutral
• Volatilidad 20d: 0.0145

Interpretación:
- El precio actual está por encima de ambas medias móviles, indicando tendencia alcista
- RSI en 58.3 está en zona neutral (ni sobrecompra ni sobreventa)
- La volatilidad del 1.45% es moderada
- La SMA 20 > SMA 50 confirma momentum alcista a corto plazo
```

---

## 📰 Consultar Noticias

**Tu pregunta:**
> ¿Qué noticias recientes hay sobre el IBEX35?

**Respuesta de Claude:**
```
📰 Últimas 5 noticias de IBEX35:

1. El IBEX 35 cierra plano tras tocar máximos del año
   📅 2025-12-03T16:30:00
   🔗 https://example.com/news1

2. Telefónica lidera las ganancias del selectivo español
   📅 2025-12-03T14:15:00
   🔗 https://example.com/news2

3. Análisis: ¿Seguirá el IBEX su racha alcista?
   📅 2025-12-03T10:00:00
   🔗 https://example.com/news3

[...]
```

---

## 🔄 Actualizar Datos

**Tu pregunta:**
> Actualiza los datos del IBEX35 del último mes y hazme una predicción

**Respuesta de Claude:**
```
✅ Datos actualizados para IBEX35:

• Precios: 21 filas actualizadas
• Indicadores: 21 filas calculadas
• Período: 1mo

Ahora voy a hacer la predicción...

🤖 Predicción ML para IBEX35:
[... predicción actualizada con datos frescos]
```

---

## 📋 Resumen Completo

**Tu pregunta:**
> Dame el análisis completo del S&P 500 para hoy

**Respuesta de Claude:**
```
📊 Resumen Diario - S&P 500

Cierre de ^GSPC el 03/12/2025: 5,908.42 puntos (+23.45, +0.40%).

Señal simple: señal alcista (+1). 
Señal ensemble: señal alcista (+1).

Indicadores técnicos: SMA20 ≈ 5,876.34, SMA50 ≈ 5,798.21, RSI14 ≈ 62.3, Volatilidad 20 días ≈ 0.0128.

Noticias recientes:
  - Fed mantiene tipos de interés sin cambios
  - Sector tecnológico impulsa ganancias en Wall Street
  - Datos de empleo superan expectativas
  [...]
```

---

## 🔍 Análisis Comparativo

**Tu pregunta:**
> Compara IBEX35, S&P500 y NASDAQ. ¿Cuál está mejor para invertir según tus modelos?

**Claude puede:**
1. Usar `get_prediction` para cada mercado
2. Usar `get_indicators` para comparar técnicos
3. Analizar las señales y dar una recomendación fundamentada

---

## ⚡ Comandos Rápidos

```
"Precio del IBEX" 
→ get_market_price

"Predicción ML del SP500"
→ get_prediction

"Indicadores del NASDAQ"
→ get_indicators

"Noticias del NIKKEI"
→ get_news

"Actualiza datos del IBEX último mes"
→ update_market_data

"Resumen diario del SP500"
→ get_daily_summary

"Valida predicciones de ayer"
→ validate_predictions
```

---

## 🎯 Casos de Uso Avanzados

### 1. Análisis Pre-Trading
> "Es lunes por la mañana. Dame un análisis completo del IBEX35, S&P500 y NASDAQ con predicciones, indicadores y noticias para decidir qué comprar esta semana."

### 2. Validación de Modelos
> "Valida las predicciones de ayer y dime qué modelo tuvo mejor accuracy"

### 3. Alertas de Oportunidades
> "Muéstrame todos los mercados con RSI en sobreventa (RSI < 30) y señal de compra ML"

### 4. Análisis de Noticias
> "Dame las noticias del IBEX de los últimos 3 días y analiza el sentiment general"

### 5. Estrategia de Trading
> "Según los indicadores técnicos y las predicciones ML, ¿debería hacer una estrategia long o short en el S&P500?"

---

## 💡 Tips

1. **Sé específico:** "Predicción ML del IBEX35" es mejor que "¿Cómo está la bolsa?"

2. **Combina herramientas:** Claude puede usar múltiples herramientas para darte análisis completos

3. **Pregunta por explicaciones:** "Explícame por qué los modelos sugieren venta"

4. **Pide contexto histórico:** "¿Cómo han performado las predicciones la última semana?"

5. **Usa lenguaje natural:** No necesitas conocer los nombres exactos de las herramientas

---

## 🚨 Limitaciones

- Los datos están limitados a lo que hay en la base de datos
- Las predicciones son indicativas, no consejos de inversión
- Los mercados deben estar en la base de datos (IBEX35, SP500, NASDAQ, NIKKEI)
- Los precios se actualizan cuando ejecutas `update_market_data` o vía workflows n8n

---

## 🎓 Aprendiendo con Claude

Puedes preguntarle a Claude sobre el sistema mismo:

> "¿Cómo funcionan tus modelos de Machine Learning?"
> "¿Qué significa RSI y cómo interpretarlo?"
> "¿Qué es un ensemble de modelos?"
> "Explícame la diferencia entre SMA 20 y SMA 50"

Claude tiene contexto sobre todo el sistema y puede explicarte conceptos técnicos de trading e IA.
