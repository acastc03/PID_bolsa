# 🤖 Guía Completa: Integración con Claude

## 📋 Tabla de Contenidos

1. [¿Qué acabamos de crear?](#qué-acabamos-de-crear)
2. [Instalación paso a paso](#instalación-paso-a-paso)
3. [Cómo funciona](#cómo-funciona)
4. [Otras formas de integrar con Claude](#otras-formas-de-integrar)
5. [Comparativa de opciones](#comparativa-de-opciones)

---

## 🎯 ¿Qué acabamos de crear?

He creado un **servidor MCP (Model Context Protocol)** que permite a Claude Desktop acceder directamente a tu sistema de predicción de mercados financieros.

### ¿Qué significa esto?

**ANTES:** 
- Tenías que llamar manualmente a tu API REST (`http://localhost:8080/...`)
- Copiar/pegar resultados para mostrarle a Claude
- Claude no podía acceder a tus datos directamente

**AHORA:**
- Claude Desktop puede consultar directamente tu base de datos
- Claude puede ejecutar predicciones ML automáticamente
- Puedes hablar con Claude naturalmente: "¿Cuál es el precio del IBEX35?"
- Claude tiene acceso a 7 herramientas especializadas

---

## 🚀 Instalación Paso a Paso

### Opción A: Instalación Automática (Recomendada)

```bash
# 1. Asegúrate de que tu base de datos está corriendo
cd /Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa
docker-compose up -d db

# 2. Ejecuta el script de instalación
./mcp_server_claude/install.sh

# 3. Reinicia Claude Desktop
# Cierra completamente la app y vuelve a abrirla
```

### Opción B: Instalación Manual

#### Paso 1: Instalar dependencias
```bash
cd mcp_server_claude
pip3 install -r requirements.txt
```

#### Paso 2: Configurar Claude Desktop

**En macOS:**
```bash
# Edita este archivo:
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Pega esta configuración:**
```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "python3",
      "args": [
        "/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa/mcp_server_claude/server.py"
      ],
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "15433",
        "DB_NAME": "indices",
        "DB_USER": "finanzas",
        "DB_PASS": "finanzas_pass",
        "PYTHONPATH": "/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa"
      }
    }
  }
}
```

⚠️ **IMPORTANTE:** Cambia `/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa` por la ruta real de tu proyecto.

#### Paso 3: Reiniciar Claude Desktop

1. Cierra completamente Claude Desktop (Cmd+Q)
2. Vuelve a abrir Claude Desktop
3. El servidor MCP se iniciará automáticamente en segundo plano

---

## 🔍 Cómo Funciona

### Arquitectura

```
┌─────────────────┐
│  Claude Desktop │
│   (interfaz)    │
└────────┬────────┘
         │ MCP Protocol
         │ (stdio)
         ▼
┌─────────────────┐
│   MCP Server    │
│  (server.py)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   PostgreSQL    │◄─────┤ API FastAPI  │
│  (tus datos)    │      │ (opcional)   │
└─────────────────┘      └──────────────┘
```

### Flujo de una consulta

1. **Tú escribes en Claude:** "¿Cuál es el precio del IBEX35?"

2. **Claude identifica:** Necesito usar la herramienta `get_market_price`

3. **MCP Server ejecuta:**
   ```python
   symbol = resolve_symbol("IBEX35")  # → "^IBEX"
   result = get_latest_price(symbol)
   ```

4. **Consulta a PostgreSQL:**
   ```sql
   SELECT date, close, open, high, low, volume
   FROM prices
   WHERE symbol = '^IBEX'
   ORDER BY date DESC
   LIMIT 1
   ```

5. **Claude recibe el resultado** y te lo muestra formateado

---

## 🛠️ Herramientas Disponibles

Claude Desktop ahora tiene acceso a estas 7 herramientas:

| Herramienta | Descripción | Ejemplo |
|-------------|-------------|---------|
| `get_market_price` | Último precio OHLCV | "Precio del IBEX35" |
| `get_prediction` | Predicción ML (7 modelos) | "Predicción del SP500" |
| `get_indicators` | SMA, RSI, Volatilidad | "Indicadores del NASDAQ" |
| `get_news` | Últimas noticias | "Noticias del NIKKEI" |
| `update_market_data` | Actualizar desde Yahoo | "Actualiza datos del IBEX" |
| `get_daily_summary` | Resumen completo | "Resumen del SP500" |
| `validate_predictions` | Validar predicciones | "Valida predicciones de ayer" |

---

## 💡 Otras Formas de Integrar con Claude

### Opción 1: MCP Server (Lo que hemos hecho) ⭐ MEJOR

**Ventajas:**
- ✅ Integración nativa con Claude Desktop
- ✅ Acceso directo a la base de datos
- ✅ Conversación natural
- ✅ Claude puede combinar múltiples herramientas
- ✅ Sin copiar/pegar
- ✅ Actualización automática de datos

**Desventajas:**
- ❌ Solo funciona con Claude Desktop (no web)
- ❌ Requiere configuración inicial

**Mejor para:** Uso diario, análisis interactivo, research

---

### Opción 2: API de Anthropic + Tu API REST

```python
import anthropic

client = anthropic.Anthropic(api_key="tu-api-key")

# Obtener datos de tu API
import requests
price = requests.get("http://localhost:8080/update_prices?market=ibex35").json()

# Enviar a Claude
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"Analiza estos datos: {price}"
    }]
)
```

**Ventajas:**
- ✅ Programático (scripts, automatización)
- ✅ Control total del flujo
- ✅ Puede ejecutarse en servidor

**Desventajas:**
- ❌ Necesitas API key de pago de Anthropic
- ❌ No es conversacional
- ❌ Código manual para cada consulta

**Mejor para:** Automatización, reportes batch, integración en pipelines

---

### Opción 3: Copiar/Pegar Manual

```bash
# 1. Obtener datos
curl http://localhost:8080/daily_summary?market=ibex35

# 2. Copiar resultado

# 3. Pegar en Claude Desktop o claude.ai
"Analiza este resumen: [pegar JSON]"
```

**Ventajas:**
- ✅ Cero configuración
- ✅ Funciona con Claude web

**Desventajas:**
- ❌ Tedioso y manual
- ❌ Propenso a errores
- ❌ No escalable

**Mejor para:** Consultas ocasionales, pruebas rápidas

---

### Opción 4: Prompt Caching con API

```python
# Cargar datos grandes una vez
system_prompt = """
Eres un analista financiero. Tienes acceso a estos datos históricos:
[... miles de líneas de datos ...]
"""

# Usar prompt caching para no pagar por los datos cada vez
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system=[{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[...]
)
```

**Ventajas:**
- ✅ Eficiente para datasets grandes
- ✅ Ahorra costos si reutilizas contexto
- ✅ Programático

**Desventajas:**
- ❌ Requiere API key
- ❌ Datos estáticos (no actualización en tiempo real)
- ❌ Complejidad adicional

**Mejor para:** Análisis de datasets grandes y estáticos

---

## 📊 Comparativa de Opciones

| Característica | MCP Server | API Anthropic | Copiar/Pegar | Prompt Cache |
|----------------|------------|---------------|--------------|--------------|
| **Facilidad de uso** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Conversacional** | ✅ Sí | ❌ No | ✅ Sí | ❌ No |
| **Tiempo real** | ✅ Sí | ✅ Sí | ✅ Sí | ❌ No |
| **Costo** | 💰 Gratis | 💰💰 Pago | 💰 Gratis | 💰💰 Pago |
| **Automatizable** | ⚠️ Limitado | ✅ Sí | ❌ No | ✅ Sí |
| **Requiere config** | ⚠️ Una vez | ✅ Siempre | ❌ No | ✅ Siempre |
| **Escalabilidad** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |

### 🏆 Recomendación

**Para ti, Gonzalo:**

1. **Uso diario y análisis:** MCP Server ⭐ (lo que acabamos de crear)
2. **Automatización futura:** API Anthropic + tu FastAPI
3. **Consultas ocasionales:** Copiar/Pegar

**Combinación ideal:**
- MCP Server para trabajo diario en Claude Desktop
- Tu API FastAPI sigue funcionando para n8n y automatización
- Ambos acceden a la misma base de datos

---

## 🧪 Probar la Integración

### 1. Verificar que el servidor está configurado

```bash
# Ver la configuración
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

### 2. Verificar la base de datos

```bash
# PostgreSQL debe estar corriendo
docker-compose ps

# Debe mostrar: db_finanzas ... Up
```

### 3. Probar en Claude Desktop

Abre Claude Desktop y escribe:

```
Hola! ¿Puedes ver las herramientas de finance-predictor?
```

Si está configurado correctamente, Claude responderá que tiene acceso a herramientas como `get_market_price`, etc.

### 4. Primera consulta real

```
¿Cuál es el precio actual del IBEX35?
```

Claude debería:
1. Usar `get_market_price`
2. Consultar tu base de datos
3. Mostrarte el resultado formateado

---

## 🐛 Solución de Problemas

### Claude no ve las herramientas

1. **Verificar ruta absoluta en config:**
   ```bash
   # Debe ser ruta completa, no relativa
   "/Users/gonzalo/Desktop/..."  # ✅ Correcto
   "~/Desktop/..."               # ❌ Incorrecto
   ```

2. **Ver logs de Claude:**
   ```bash
   # macOS
   tail -f ~/Library/Logs/Claude/mcp*.log
   ```

3. **Reiniciar completamente Claude:**
   - Cmd+Q (no solo cerrar ventana)
   - Volver a abrir

### Error: "No module named 'mcp'"

```bash
pip3 install mcp
```

### Error: "Cannot connect to database"

```bash
# Verificar PostgreSQL
docker-compose up -d db

# Verificar puerto
lsof -i :15433
```

### Error: "PYTHONPATH not set"

En la configuración JSON, asegúrate de que `PYTHONPATH` apunta al directorio raíz:
```json
"PYTHONPATH": "/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa"
```

---

## 📚 Recursos

- [Documentación MCP](mcp_server_claude/README.md)
- [Ejemplos de uso](mcp_server_claude/EJEMPLOS.md)
- [Model Context Protocol Spec](https://modelcontextprotocol.io/)
- [API de tu proyecto](http://localhost:8080/docs)

---

## 🎯 Próximos Pasos

1. ✅ **Instalar:** Ejecuta `./mcp_server_claude/install.sh`
2. ✅ **Probar:** Pregunta a Claude sobre precios
3. 📊 **Explorar:** Lee los [ejemplos](mcp_server_claude/EJEMPLOS.md)
4. 🚀 **Usar:** Integra en tu workflow diario

---

## 💬 Ejemplo Real Completo

**Conversación con Claude Desktop:**

```
Tú: Hola! Dame un análisis completo del mercado español

Claude: [Usa get_market_price, get_indicators, get_news, get_prediction]

📊 Análisis del IBEX35 - 03/12/2025

PRECIO:
• Cierre: 11,543.21 (+0.45%)
• Volumen: 234M

INDICADORES TÉCNICOS:
• SMA 20: 11,498 (precio > SMA → tendencia alcista)
• RSI 14: 58.3 (neutral, no sobrecompra/sobreventa)
• Volatilidad: 1.2% (moderada)

PREDICCIÓN ML:
• Señal Ensemble: 🟢 COMPRA
• 6 de 7 modelos sugieren alza
• Precio objetivo: ~11,580

NOTICIAS RECIENTES:
1. Telefónica lidera ganancias del selectivo
2. IBEX toca máximos del año
[...]

RECOMENDACIÓN:
Los indicadores técnicos y modelos ML coinciden en tendencia
alcista a corto plazo. RSI en zona segura permite entrada.
Considerar soporte en 11,475.
```

**Todo esto sin que escribas una sola línea de código o hagas ninguna API call manual!** 🎉

---

¿Necesitas ayuda con la instalación o tienes dudas? ¡Pregúntame!
