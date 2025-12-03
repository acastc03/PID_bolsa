# 🤖 Integración con Claude Desktop

Este directorio contiene el servidor MCP (Model Context Protocol) para integrar el sistema de predicción de mercados con Claude Desktop.

## 🚀 Instalación

### 1. Instalar dependencias

```bash
cd mcp_server_claude
pip install -r requirements.txt
```

### 2. Configurar Claude Desktop

Edita el archivo de configuración de Claude Desktop:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%/Claude/claude_desktop_config.json`

Añade esta configuración:

#### Opción A: Docker (Recomendado) 🐳

**macOS/Linux:**
```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "/ruta/absoluta/a/tu/proyecto/mcp_server_claude/run_docker_optimized.sh",
      "args": []
    }
  }
}
```

**Ejemplo macOS:**
```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa/mcp_server_claude/run_docker_optimized.sh",
      "args": []
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "bash",
      "args": [
        "C:\\Users\\TuUsuario\\PID_bolsa\\mcp_server_claude\\run_docker_optimized.sh"
      ]
    }
  }
}
```

💡 **Windows**: Necesitas WSL2 o Git Bash instalado para ejecutar scripts `.sh`

✅ **Ventajas:**
- Aislamiento completo
- No requiere instalar dependencias localmente
- Funciona igual en cualquier máquina
- Mismo entorno en desarrollo y producción

#### Opción B: Python Directo 🐍

**macOS/Linux:**
```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "python3",
      "args": [
        "/ruta/absoluta/a/tu/proyecto/mcp_server_claude/server.py"
      ],
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "15433",
        "DB_NAME": "indices",
        "DB_USER": "finanzas",
        "DB_PASS": "finanzas_pass",
        "PYTHONPATH": "/ruta/absoluta/a/tu/proyecto"
      }
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "python",
      "args": [
        "C:\\Users\\TuUsuario\\PID_bolsa\\mcp_server_claude\\server.py"
      ],
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "15433",
        "DB_NAME": "indices",
        "DB_USER": "finanzas",
        "DB_PASS": "finanzas_pass",
        "PYTHONPATH": "C:\\Users\\TuUsuario\\PID_bolsa"
      }
    }
  }
}
```

⚠️ **IMPORTANTE:** 
- **macOS/Linux**: Usa rutas con `/` (ejemplo: `/Users/usuario/proyecto`)
- **Windows**: Usa rutas con `\\` (ejemplo: `C:\\Users\\usuario\\proyecto`)
- **Windows**: Usa `python` en lugar de `python3`

📖 **Más información:** Ver [DOCKER_SETUP.md](./DOCKER_SETUP.md) para comparación detallada.

### 3. Reiniciar Claude Desktop

Cierra completamente Claude Desktop y vuelve a abrirlo. El servidor MCP se iniciará automáticamente.

## 🛠️ Herramientas Disponibles

Una vez configurado, Claude Desktop tendrá acceso a estas herramientas:

### 📊 `get_market_price`
Obtiene el último precio de un mercado (IBEX35, SP500, NASDAQ, NIKKEI).

**Ejemplo de uso en Claude:**
> "¿Cuál es el precio actual del IBEX35?"

### 🤖 `get_prediction`
Obtiene predicciones ML usando ensemble de 7 modelos.

**Ejemplo:**
> "Dame la predicción de Machine Learning para el S&P 500"

### 📈 `get_indicators`
Obtiene indicadores técnicos (SMA, RSI, volatilidad).

**Ejemplo:**
> "Muéstrame los indicadores técnicos del NASDAQ"

### 📰 `get_news`
Obtiene las últimas noticias del mercado.

**Ejemplo:**
> "¿Qué noticias recientes hay sobre el IBEX35?"

### 🔄 `update_market_data`
Actualiza datos desde Yahoo Finance.

**Ejemplo:**
> "Actualiza los datos del IBEX35 del último mes"

### 📋 `get_daily_summary`
Obtiene un resumen completo del día.

**Ejemplo:**
> "Dame el resumen diario del S&P 500"

### ✅ `validate_predictions`
Valida predicciones del día anterior.

**Ejemplo:**
> "Valida las predicciones de ayer"

## 🧪 Probar el Servidor

Puedes probar el servidor directamente:

```bash
# Asegúrate de que PostgreSQL está corriendo
docker-compose up -d db

# Ejecutar el servidor en modo de prueba
cd /Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa
export PYTHONPATH=$PWD
export DB_HOST=localhost
export DB_PORT=15433
export DB_NAME=indices
export DB_USER=finanzas
export DB_PASS=finanzas_pass

python3 mcp_server_claude/server.py
```

El servidor esperará comandos en stdin (MCP usa stdio para comunicación).

## 🔍 Debugging

Si Claude no ve las herramientas:

1. **Verifica los logs:** Claude Desktop muestra errores en la interfaz
2. **Comprueba las rutas:** Asegúrate de que la ruta en `args` es absoluta y correcta
3. **Verifica la base de datos:** PostgreSQL debe estar corriendo en el puerto 15433
4. **Comprueba las variables de entorno:** Deben coincidir con tu `.env`

Ver logs en:
- macOS: `~/Library/Logs/Claude/`
- Windows: `%APPDATA%/Claude/logs/`

## 💡 Ejemplos de Conversación con Claude

Una vez configurado, puedes hablar con Claude naturalmente:

```
Usuario: "¿Qué tal está el IBEX35 hoy?"
Claude: [Usa get_market_price y get_indicators]
        "El IBEX35 cerró a 11,543.21 puntos..."

Usuario: "¿Debería comprar o vender según tus modelos?"
Claude: [Usa get_prediction]
        "Según el ensemble de 7 modelos ML, la señal es COMPRA (+1)..."

Usuario: "Dame el análisis completo con noticias"
Claude: [Usa get_daily_summary y get_news]
        "Aquí está el resumen completo..."
```

## 🔧 Troubleshooting

### Error: "No module named 'mcp'"
```bash
pip install mcp
```

### Error: "No se puede conectar a la base de datos"
Verifica que PostgreSQL esté corriendo:
```bash
docker-compose ps
```

### Error: "PYTHONPATH no configurado"
Asegúrate de que el `PYTHONPATH` en la configuración apunta al directorio raíz del proyecto.

## 📚 Documentación

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Claude Desktop Documentation](https://www.anthropic.com/claude)
- [Documentación del proyecto](../README.md)
