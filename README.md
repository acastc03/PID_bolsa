# 📈 PID Bolsa - Sistema de Predicción de Mercados Financieros

Sistema completo de ingesta, procesamiento y predicción de datos financieros para índices bursátiles (IBEX35, S&P500, NASDAQ, NIKKEI) utilizando Machine Learning y automatización de workflows.

## 🎯 Características Principales

- **📊 Ingesta Automática de Datos**: Descarga históricos de precios vía yfinance
- **📰 Análisis de Noticias**: Recopilación y análisis de sentiment de noticias financieras
- **🤖 Predicción ML**: Ensemble de modelos (LinearRegression, Prophet, XGBoost, LightGBM, CatBoost)
- **📈 Indicadores Técnicos**: SMA, RSI, Volatilidad
- **🔄 Automatización**: Workflows diarios con n8n
- **🐳 Dockerizado**: Despliegue completo con Docker Compose
- **📊 Base de Datos**: PostgreSQL para almacenamiento persistente
- **🔍 API REST**: FastAPI para acceso a datos y predicciones

## 🏗️ Arquitectura

```
┌─────────────┐
│    n8n      │ ──► Orquestación de workflows diarios
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  MCP Server │ ──► API FastAPI (predicciones ML)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ PostgreSQL  │ ──► Almacenamiento de datos
└─────────────┘
       │
       ▼
┌─────────────┐
│   Adminer   │ ──► Gestión de BD (interfaz web)
└─────────────┘
```

## 📦 Componentes

### 1. Base de Datos (PostgreSQL)
- **Puerto**: 15433 (configurable en `.env`)
- **Tablas**:
  - `prices`: Datos históricos OHLCV
  - `indicators`: Indicadores técnicos (SMA, RSI, etc.)
  - `signals`: Señales de trading (+1, 0, -1)
  - `news`: Noticias con análisis de sentiment
  - `ml_predictions`: Predicciones diarias de modelos ML

### 0. 🤖 Claude Desktop Integration (NUEVO)
- **Servidor MCP** en `mcp_server_claude/`
- Permite a Claude Desktop acceder a todas las funcionalidades
- 7 herramientas disponibles para análisis conversacional
- [Ver guía de integración](mcp_server_claude/README.md)

### 2. MCP Server (FastAPI)
API REST para:
- Actualización de precios y noticias
- Cálculo de indicadores técnicos
- Entrenamiento y predicción de modelos ML
- Validación de predicciones históricas
- Reportes diarios

### 3. n8n (Automatización)
- **Puerto**: 5678
- **Credenciales**: admin / admin123
- Workflows para:
  - Ingesta diaria de datos
  - Cálculo de indicadores
  - Reentrenamiento de modelos
  - Generación de reportes

### 4. Adminer (Gestión BD)
- **Puerto**: 8081
- Interfaz web para consultar y gestionar la base de datos

## 🚀 Instalación y Uso

### Prerrequisitos

- Docker y Docker Compose
- Python 3.11+ (para desarrollo local)
- Git

### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd PID_bolsa
```

### 2. Configurar Variables de Entorno

Crear o editar el archivo `.env`:

```env
# Puertos expuestos
POSTGRES_PORT=15433
N8N_PORT=5678
MCP_PORT=8080

# Configuración de BD
POSTGRES_USER=finanzas
POSTGRES_PASSWORD=finanzas_pass
POSTGRES_DB=indices

# Base de datos para MCP
MCP_DB_NAME=indices
```

### 3. Iniciar los Servicios

```bash
docker-compose up -d
```

Esto iniciará:
- PostgreSQL en `localhost:15433`
- n8n en `http://localhost:5678`
- MCP Server en `http://localhost:8080`
- Adminer en `http://localhost:8081`

### 4. Verificar el Estado

```bash
# Ver logs
docker-compose logs -f

# Verificar que todos los servicios estén corriendo
docker-compose ps

# Probar la API
curl http://localhost:8080/health
```

## 📚 Uso de la API

### Documentación Interactiva

Acceder a la documentación Swagger:
```
http://localhost:8080/docs
```

### Endpoints Principales

#### 🔄 ETL - Ingesta de Datos

```bash
# Actualizar precios del IBEX35 (último mes)
curl "http://localhost:8080/update_prices?market=ibex35&period=1mo"

# Actualizar noticias para múltiples mercados
curl "http://localhost:8080/update_news?markets=IBEX35,SP500&days=7"
```

#### 📊 ETL - Procesamiento

```bash
# Calcular indicadores técnicos
curl "http://localhost:8080/compute_indicators?market=ibex35"

# Generar señales de trading
curl "http://localhost:8080/compute_signals?market=ibex35"
```

#### 🤖 Machine Learning

```bash
# Predicción simple (reglas)
curl "http://localhost:8080/predecir_simple?symbol=^IBEX"

# Predicción ensemble (ML)
curl "http://localhost:8080/predecir_ensemble?symbol=^IBEX"

# Forzar reentrenamiento de modelos
curl "http://localhost:8080/retrain_models?symbol=^IBEX"

# Validar predicciones de ayer
curl -X POST "http://localhost:8080/validate_predictions"

# Validar predicciones de una fecha específica
curl -X POST "http://localhost:8080/validate_predictions?date_str=2025-11-25"
```

#### 📈 Reporting

```bash
# Resumen diario del mercado
curl "http://localhost:8080/daily_summary?market=ibex35"

# Información de modelos guardados
curl "http://localhost:8080/model_info?symbol=^IBEX"
```

## 🤖 Integración con Claude Desktop

### Configuración Rápida

1. **Instalar dependencias del servidor MCP:**
```bash
cd mcp_server_claude
pip install -r requirements.txt
```

2. **Configurar Claude Desktop:**

Editar `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "finance-predictor": {
      "command": "python3",
      "args": [
        "/ruta/completa/al/proyecto/PID_bolsa/mcp_server_claude/server.py"
      ],
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "15433",
        "DB_NAME": "indices",
        "DB_USER": "finanzas",
        "DB_PASS": "finanzas_pass",
        "PYTHONPATH": "/ruta/completa/al/proyecto/PID_bolsa"
      }
    }
  }
}
```

3. **Reiniciar Claude Desktop**

Ahora puedes preguntarle a Claude cosas como:
- "¿Cuál es el precio actual del IBEX35?"
- "Dame la predicción ML para el S&P 500"
- "Muéstrame el resumen diario completo"

**📖 [Guía completa de integración con Claude](mcp_server_claude/README.md)**

---

## 🔧 Desarrollo Local

### Instalar Dependencias

```bash
# Crear entorno virtual
python3 -m venv PID
source PID/bin/activate

# Instalar dependencias
pip install -r mcp_server/requirements.txt
```

### Ejecutar el Servidor MCP Localmente

```bash
cd mcp_server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Script de Descarga Manual

```bash
python download_ibex.py
```

Esto descargará los datos del IBEX35 en `./data/^IBEX_prices.csv`.

## 📊 Modelos de Machine Learning

El sistema utiliza un **ensemble** de 5 modelos:

1. **Linear Regression**: Modelo base de regresión lineal
2. **Prophet**: Modelo de series temporales de Facebook
3. **XGBoost**: Gradient boosting optimizado
4. **LightGBM**: Gradient boosting ligero y rápido
5. **CatBoost**: Gradient boosting con manejo automático de categorías

### Características (Features)

- Precios: Open, High, Low, Close, Volume
- Indicadores técnicos: SMA(20), SMA(50), RSI(14), Volatilidad(20)
- Features temporales: Día de la semana, mes, retornos previos

### Señales de Predicción

- **+1**: Señal de compra (el precio subirá)
- **0**: Mantener posición (sin movimiento significativo)
- **-1**: Señal de venta (el precio bajará)

### Votación Ensemble

La señal final se determina por mayoría simple de los 5 modelos.

## 🗂️ Estructura del Proyecto

```
PID_bolsa/
├── docker-compose.yml          # Orquestación de servicios
├── .env                        # Variables de entorno
├── download_ibex.py           # Script de descarga manual
├── requests.http              # Ejemplos de peticiones HTTP
├── data/                      # Datos persistentes
│   ├── db/                    # Volumen PostgreSQL
│   └── models/                # Modelos ML guardados
├── db-init/                   # Scripts de inicialización BD
│   ├── 01_init.sql           # Tablas principales
│   └── 02_ml_predictions.sql # Tabla de predicciones
├── mcp_server/               # API FastAPI
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py          # Endpoints FastAPI
│   └── scripts/             # Módulos de lógica
│       ├── assets.py        # Resolución de símbolos
│       ├── fetch_data.py    # Descarga de precios
│       ├── indicators.py    # Cálculo de indicadores
│       ├── models.py        # Modelos ML
│       ├── news.py          # Scraping de noticias
│       ├── save_predictions.py    # Persistencia de predicciones
│       ├── validate_predictions.py # Validación de modelos
│       ├── reporting.py     # Generación de reportes
│       └── model_storage.py # Gestión de modelos
├── n8n/                      # Datos de n8n
└── PID/                      # Entorno virtual Python
```

## 🔄 Workflow Diario Recomendado

### Configurar en n8n o ejecutar manualmente:

1. **08:00** - Actualizar precios de mercados
   ```bash
   curl "http://localhost:8080/update_prices?market=ibex35&period=5d"
   ```

2. **08:05** - Descargar noticias recientes
   ```bash
   curl "http://localhost:8080/update_news?markets=IBEX35,SP500&days=1"
   ```

3. **08:10** - Calcular indicadores técnicos
   ```bash
   curl "http://localhost:8080/compute_indicators?market=ibex35"
   ```

4. **08:15** - Generar señales de trading
   ```bash
   curl "http://localhost:8080/compute_signals?market=ibex35"
   ```

5. **08:20** - Reentrenar modelos y hacer predicción
   ```bash
   curl "http://localhost:8080/retrain_models?symbol=^IBEX"
   ```

6. **08:25** - Validar predicciones del día anterior
   ```bash
   curl -X POST "http://localhost:8080/validate_predictions"
   ```

7. **08:30** - Generar reporte diario
   ```bash
   curl "http://localhost:8080/daily_summary?market=ibex35"
   ```

## 🛠️ Mantenimiento

### Ver Logs

```bash
# Logs de todos los servicios
docker-compose logs -f

# Logs de un servicio específico
docker-compose logs -f mcp
docker-compose logs -f db
docker-compose logs -f n8n
```

### Backup de la Base de Datos

```bash
docker exec db_finanzas pg_dump -U finanzas indices > backup_$(date +%Y%m%d).sql
```

### Restaurar Backup

```bash
docker exec -i db_finanzas psql -U finanzas indices < backup_20251126.sql
```

### Limpiar Modelos Antiguos

Los modelos se limpian automáticamente manteniendo los últimos 7 días. Para limpiar manualmente:

```bash
curl "http://localhost:8080/retrain_models?symbol=^IBEX"
```

### Reiniciar Servicios

```bash
# Reiniciar todos los servicios
docker-compose restart

# Reiniciar un servicio específico
docker-compose restart mcp
```

### Detener y Eliminar Todo

```bash
docker-compose down

# Eliminar también los volúmenes (⚠️ BORRA TODOS LOS DATOS)
docker-compose down -v
```

## 📊 Gestión de Base de Datos

### Acceder con Adminer

1. Ir a `http://localhost:8081`
2. Ingresar credenciales:
   - **Sistema**: PostgreSQL
   - **Servidor**: db
   - **Usuario**: finanzas
   - **Contraseña**: finanzas_pass
   - **Base de datos**: indices

### Consultas Útiles

```sql
-- Ver últimos precios
SELECT * FROM prices WHERE symbol = '^IBEX' ORDER BY date DESC LIMIT 10;

-- Ver indicadores recientes
SELECT * FROM indicators WHERE symbol = '^IBEX' ORDER BY date DESC LIMIT 10;

-- Ver señales generadas
SELECT * FROM signals WHERE symbol = '^IBEX' ORDER BY date DESC LIMIT 10;

-- Ver predicciones ML con errores
SELECT 
    prediction_date,
    model_name,
    predicted_value,
    true_value,
    error_abs,
    CASE 
        WHEN true_value IS NOT NULL 
        THEN ABS(error_abs / true_value) * 100 
    END as error_percent
FROM ml_predictions 
WHERE symbol = '^IBEX' 
    AND true_value IS NOT NULL
ORDER BY prediction_date DESC, model_name;

-- Comparar rendimiento de modelos
SELECT 
    model_name,
    COUNT(*) as predictions,
    AVG(error_abs) as avg_error,
    AVG(ABS(error_abs / true_value) * 100) as avg_error_percent
FROM ml_predictions
WHERE symbol = '^IBEX' 
    AND true_value IS NOT NULL
GROUP BY model_name
ORDER BY avg_error;

-- Ver noticias recientes
SELECT * FROM news WHERE symbol = '^IBEX' ORDER BY published_at DESC LIMIT 10;
```

## 🔐 Seguridad

⚠️ **IMPORTANTE**: Este proyecto es para uso educativo/desarrollo.

Para producción:
- Cambiar credenciales por defecto en `.env`
- Usar secrets de Docker en lugar de variables de entorno
- Configurar HTTPS con certificados SSL
- Implementar autenticación JWT en la API
- Configurar firewall y limitar acceso a puertos

## 🐛 Solución de Problemas

### Error: "Puerto ya en uso"

Cambiar los puertos en `.env`:
```env
POSTGRES_PORT=15434
N8N_PORT=5679
MCP_PORT=8081
```

### Error: "No se puede conectar a la base de datos"

1. Verificar que PostgreSQL esté corriendo:
   ```bash
   docker-compose ps
   ```

2. Verificar logs:
   ```bash
   docker-compose logs db
   ```

3. Reiniciar el servicio:
   ```bash
   docker-compose restart db
   ```

### Error: "Modelos ML no se entrenan"

Verificar que haya suficientes datos:
```sql
SELECT COUNT(*) FROM prices WHERE symbol = '^IBEX';
SELECT COUNT(*) FROM indicators WHERE symbol = '^IBEX';
```

Se necesitan al menos 60 días de datos históricos para entrenar correctamente.

### Limpiar y Reiniciar

```bash
# Detener todo
docker-compose down

# Eliminar volúmenes (⚠️ borra datos)
docker-compose down -v

# Reconstruir imágenes
docker-compose build --no-cache

# Iniciar de nuevo
docker-compose up -d
```

## 📈 Mercados Soportados

| Mercado | Símbolo | Descripción |
|---------|---------|-------------|
| IBEX35 | ^IBEX | Índice español |
| SP500 | ^GSPC | S&P 500 (USA) |
| NASDAQ | ^IXIC | NASDAQ Composite |
| NIKKEI | ^N225 | Nikkei 225 (Japón) |

## 📄 Licencia

Este proyecto es de código abierto para uso educativo.

## 👥 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📞 Soporte

Para preguntas o problemas:
- Abrir un issue en el repositorio
- Revisar la documentación de la API en `/docs`
- Consultar los logs de los servicios

---

**Desarrollado con ❤️ para el curso de Ingeniería de Datos**
