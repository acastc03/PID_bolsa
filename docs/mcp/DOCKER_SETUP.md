# Ejecutar MCP Server con Docker

El servidor MCP puede ejecutarse tanto **directamente** como dentro de **Docker**. Aquí te explicamos ambas opciones para **macOS, Linux y Windows**.

---

## 📋 Requisitos Previos

### Todos los sistemas
- ✅ **Docker Desktop** instalado y ejecutándose
  - macOS: https://docs.docker.com/desktop/install/mac-install/
  - Windows: https://docs.docker.com/desktop/install/windows-install/
  - Linux: https://docs.docker.com/desktop/install/linux-install/
- ✅ **Claude Desktop** instalado
- ✅ **PostgreSQL** corriendo (vía docker-compose)

### Solo Windows (para scripts .sh)
- ✅ **Git Bash** (incluido con Git for Windows)
  - Descarga: https://git-scm.com/downloads
- O **WSL2** (Windows Subsystem for Linux)
  - Instalación: `wsl --install` en PowerShell como Admin

---

## 🐳 Opción 1: Docker Simple (Más lento al inicio)

Este método crea un contenedor temporal cada vez que Claude Desktop se conecta.

### Configuración Claude Desktop

#### macOS/Linux

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

#### Windows

Edita `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "C:\\Users\\TuUsuario\\PID_bolsa\\mcp_server_claude\\run_docker.bat",
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

#### macOS/Linux
```bash
cd /Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa
docker build -t mcp-finance-server:latest -f mcp_server_claude/Dockerfile .
```

#### Windows (PowerShell)
```powershell
cd C:\Users\TuUsuario\PID_bolsa
docker build -t mcp-finance-server:latest -f mcp_server_claude\Dockerfile .
```

#### Windows (CMD)
```cmd
cd C:\Users\TuUsuario\PID_bolsa
docker build -t mcp-finance-server:latest -f mcp_server_claude\Dockerfile .
```

### 2. Configuración Claude Desktop

#### macOS/Linux

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

#### Windows

Edita `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "C:\\Users\\TuUsuario\\PID_bolsa\\mcp_server_claude\\run_docker_optimized.bat",
      "args": []
    }
  }
}
```

**Ventajas:**
- ✅ Muy rápido al inicio (~1-2 segundos)
- ✅ Entorno aislado
- ✅ No descarga dependencias cada vez
- ✅ Funciona igual en Windows, macOS y Linux

**Desventajas:**
- ❌ Necesitas reconstruir la imagen si cambias el código:
  
  **macOS/Linux:**
  ```bash
  docker build -t mcp-finance-server:latest -f mcp_server_claude/Dockerfile .
  ```
  
  **Windows:**
  ```cmd
  docker build -t mcp-finance-server:latest -f mcp_server_claude\Dockerfile .
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

### macOS/Linux
1. Edita `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Cambia el `command` por el método que prefieras
3. Reinicia Claude Desktop (Cmd+Q y vuelve a abrir)

### Windows
1. Edita `%APPDATA%\Claude\claude_desktop_config.json`
2. Cambia el `command` por el método que prefieras
3. Reinicia Claude Desktop (Alt+F4 y vuelve a abrir)

---

## 🐛 Troubleshooting Docker

### Error: "Cannot connect to database"

Asegúrate de que PostgreSQL está corriendo:

**macOS/Linux:**
```bash
docker ps | grep db_finanzas
```

**Windows (PowerShell):**
```powershell
docker ps | Select-String db_finanzas
```

Si no está corriendo:

**macOS/Linux:**
```bash
cd /Users/gonzalo/Desktop/ING.DATOS/4º/PID/PID_bolsa
docker-compose up -d db
```

**Windows:**
```cmd
cd C:\Users\TuUsuario\PID_bolsa
docker-compose up -d db
```

### Error: "Docker is not running"

- **macOS**: Abre Docker Desktop desde Applications
- **Windows**: Abre Docker Desktop desde el menú Inicio
- **Linux**: `sudo systemctl start docker`

### Error: Scripts .sh no funcionan en Windows

Asegúrate de tener instalado:
- **Git Bash** (incluido con Git for Windows): https://git-scm.com/downloads
- O **WSL2** (Windows Subsystem for Linux): https://docs.microsoft.com/en-us/windows/wsl/install

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
