# Ejecutar MCP Server con Docker

El servidor MCP puede ejecutarse tanto **directamente** como dentro de **Docker**. Aquí te explicamos ambas opciones.

---

## 🐳 Opción 1: Docker Simple (Más lento al inicio)

Este método crea un contenedor temporal cada vez que Claude Desktop se conecta.

### Configuración Claude Desktop

Edita `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa/mcp_server_claude/run_docker.sh",
      "args": []
    }
  }
}
```

**Ventajas:**
- ✅ Entorno aislado
- ✅ No afecta a tu sistema local
- ✅ Dependencias siempre consistentes

**Desventajas:**
- ❌ Más lento al inicio (~10-15 segundos)
- ❌ Descarga dependencias cada vez

---

## 🚀 Opción 2: Docker Optimizado (Recomendado)

Este método pre-construye una imagen Docker con todas las dependencias, haciéndolo mucho más rápido.

### 1. Construir la imagen Docker

```bash
cd /Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa
docker build -t mcp-finance-server:latest -f mcp_server_claude/Dockerfile .
```

### 2. Configuración Claude Desktop

Edita `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

**Ventajas:**
- ✅ Muy rápido al inicio (~1-2 segundos)
- ✅ Entorno aislado
- ✅ No descarga dependencias cada vez

**Desventajas:**
- ❌ Necesitas reconstruir la imagen si cambias el código:
  ```bash
  docker build -t mcp-finance-server:latest -f mcp_server_claude/Dockerfile .
  ```

---

## 💻 Opción 3: Ejecución Directa (Actual)

El método actual que ya tienes configurado, ejecutando Python directamente.

### Configuración Claude Desktop

```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "/opt/homebrew/opt/python@3.11/bin/python3.11",
      "args": ["/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa/mcp_server_claude/server.py"],
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "15433",
        "DB_NAME": "indices",
        "DB_USER": "finanzas",
        "DB_PASS": "finanzas_pass",
        "PYTHONPATH": "/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa",
        "VIRTUAL_ENV": "/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa/PID",
        "PATH": "/Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa/PID/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
      }
    }
  }
}
```

**Ventajas:**
- ✅ Más rápido de todas las opciones (<1 segundo)
- ✅ Fácil de debuggear
- ✅ No necesita Docker

**Desventajas:**
- ❌ Dependencias en tu sistema local
- ❌ Posibles conflictos con otras versiones

---

## 🧪 Probar las configuraciones

### Probar script Docker directamente:

```bash
# Docker simple
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | \
  /Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa/mcp_server_claude/run_docker.sh

# Docker optimizado (primero construye la imagen)
docker build -t mcp-finance-server:latest \
  -f mcp_server_claude/Dockerfile .

echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | \
  /Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa/mcp_server_claude/run_docker_optimized.sh
```

### Ver logs en Claude Desktop:

```bash
tail -f ~/Library/Logs/Claude/mcp-server-finance-predictor.log
```

---

## 📊 Comparación de Rendimiento

| Método | Tiempo inicio | Aislamiento | Mantenimiento |
|--------|--------------|-------------|---------------|
| **Directa** | <1s | ❌ | ⭐⭐ |
| **Docker Simple** | ~15s | ✅ | ⭐⭐⭐ |
| **Docker Optimizado** | ~2s | ✅ | ⭐⭐⭐⭐ |

---

## 🔄 Cambiar entre métodos

1. Edita `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Cambia el `command` por el método que prefieras
3. Reinicia Claude Desktop (Cmd+Q y vuelve a abrir)

---

## 🐛 Troubleshooting Docker

### Error: "Cannot connect to database"

Asegúrate de que PostgreSQL está corriendo:

```bash
docker ps | grep db_finanzas
```

Si no está corriendo:

```bash
cd /Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa
docker-compose up -d db
```

### Error: "Docker daemon not running"

Inicia Docker Desktop:

```bash
open -a Docker
```

### Reconstruir imagen después de cambios en el código

```bash
docker build -t mcp-finance-server:latest \
  -f mcp_server_claude/Dockerfile . --no-cache
```

---

## 💡 Recomendación

Para **desarrollo**: Usa ejecución **directa** (más rápida, fácil de debuggear)

Para **producción/demo**: Usa **Docker optimizado** (aislamiento, consistencia)
