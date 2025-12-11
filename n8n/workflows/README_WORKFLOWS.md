# 🔄 Automatizaciones n8n - Sistema PID Bolsa

## 📋 Descripción

Este directorio contiene 3 workflows automatizados para el sistema de predicción de mercados financieros, adaptados para los **3 índices globales**: IBEX35 (España), SP500 (USA), NIKKEI (Japón).

---

## 🌅 1. Morning Data Ingestion (09:00 AM)

**Archivo**: `01_morning_data_ingestion.json`

### Objetivo
Actualización matinal de datos de mercado y generación de predicciones para los 3 índices.

### Flujo de Trabajo
1. **Health Check**: Verifica que la API esté operativa
2. **Loop por cada mercado** (IBEX35, SP500, NIKKEI):
   - Actualiza precios históricos (último mes)
   - Recopila noticias financieras
   - Calcula indicadores técnicos (SMA, RSI, volatilidad)
   - Genera predicción ensemble con 7 modelos ML

### Horario
- **Lunes a Viernes**: 09:00 AM
- **Cron**: `0 9 * * 1-5`

### Endpoints Utilizados
- `GET /health` - Verificación de API
- `GET /update_prices?market={MARKET}&period=1mo`
- `GET /update_news?market={MARKET}`
- `GET /compute_indicators?market={MARKET}`
- `GET /predecir_ensemble?market={MARKET}`

---

## 🌙 2. Evening Validation & Retrain (07:00 PM)

**Archivo**: `02_evening_validation_retrain.json`

### Objetivo
Validación de predicciones del día anterior y reentrenamiento de modelos ML para los 3 índices.

### Flujo de Trabajo
1. **Health Check**: Verifica que la API esté operativa
2. **Loop por cada mercado** (^IBEX, ^GSPC, ^N225):
   - Valida predicciones de ayer comparándolas con precios reales
   - Reentrena los 7 modelos ML con datos actualizados
   - Limpia modelos antiguos (mantiene últimos 7 días)
   - Genera reporte de rendimiento de modelos (últimos 7 días)

### Horario
- **Lunes a Viernes**: 07:00 PM
- **Cron**: `0 19 * * 1-5`

### Endpoints Utilizados
- `GET /health` - Verificación de API
- `POST /validate_and_retrain?symbol={SYMBOL}` - Workflow completo
- `GET /model_performance?symbol={SYMBOL}&days=7` - Análisis de rendimiento

---

## 📧 3. Daily Email Report with Gemini (08:00 PM)

**Archivo**: `03_daily_email_report.json`

### Objetivo
Generación y envío de reporte diario por email con resumen de los 3 mercados, redactado por Gemini AI.

### Flujo de Trabajo
1. **Loop por cada mercado** (IBEX35, SP500, NIKKEI):
   - Obtiene resumen completo del día (precios, indicadores, señales, noticias, ML)
2. **Agregación de datos**: Combina información de los 3 mercados
3. **Gemini AI**: Genera reporte narrativo profesional
4. **Email**: Envía reporte formateado

### Horario
- **Lunes a Viernes**: 08:00 PM
- **Cron**: `0 20 * * 1-5`

### Endpoints Utilizados
- `GET /daily_summary?market={MARKET}&include_ml=true`

### ⚠️ Configuración Adicional Requerida

Este workflow requiere configuración manual en n8n:

#### 1. Nodo Gemini AI
Después del nodo "Prepare Data", agregar:

**Opción A: Nodo HTTP Request a Gemini**
```
Método: POST
URL: https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY
Headers:
  Content-Type: application/json
Body:
{
  "contents": [{
    "parts": [{
      "text": "Genera un reporte profesional de mercados financieros basado en estos datos: {{ $json.market_data }}"
    }]
  }]
}
```

**Opción B: Si existe nodo nativo Gemini en n8n**
- Configurar credenciales con API Key
- Modelo: `gemini-pro`
- Prompt: Ver ejemplo arriba

**Obtener API Key**: https://aistudio.google.com/

#### 2. Nodo Aggregate
Para combinar datos de los 3 mercados antes de enviar a Gemini:
- Tipo: `aggregateAllItemData`
- Formato: JSON array con los 3 summaries

#### 3. Nodo Send Email
**Opción A: Gmail (OAuth2)**
- Configurar OAuth2 en Google Cloud Console
- Agregar credenciales en n8n

**Opción B: SMTP**
```
Host: smtp.gmail.com (o tu servidor)
Port: 587
User: tu-email@example.com
Password: contraseña de aplicación
TLS: true
```

**Opción C: Servicios externos**
- SendGrid
- Mailgun
- Amazon SES

**Configuración del email**:
- **To**: tu-email@example.com
- **Subject**: `📈 Reporte Diario Mercados - {{ $now.format('DD/MM/YYYY') }}`
- **Body**: `{{ $json.gemini_response }}`
- **Format**: HTML

---

## 📥 Importación a n8n

### Método 1: Interfaz Web
1. Accede a n8n: http://localhost:5678
2. Click en **"+"** → **"Import from File"**
3. Selecciona el archivo JSON del workflow
4. Click en **"Import"**
5. Activa el workflow con el toggle

### Método 2: Copiar a contenedor
```powershell
# Copiar workflows al contenedor n8n
docker cp n8n/workflows/01_morning_data_ingestion.json n8n:/data/workflows/
docker cp n8n/workflows/02_evening_validation_retrain.json n8n:/data/workflows/
docker cp n8n/workflows/03_daily_email_report.json n8n:/data/workflows/

# Reiniciar n8n para detectar workflows
docker-compose restart n8n
```

---

## 🔧 Configuración de Horarios

Los horarios están optimizados para mercados españoles (CET/CEST):

| Workflow | Horario | Razón |
|----------|---------|-------|
| Morning Ingestion | 09:00 AM | Antes de apertura europea |
| Validation & Retrain | 07:00 PM | Después de cierre europeo y durante trading USA |
| Email Report | 08:00 PM | Final del día, tras validación |

**Ajustar para otras zonas horarias**:
Editar el `cronExpression` en cada workflow.

Ejemplos:
- **UTC**: `0 8 * * 1-5` (08:00 AM UTC)
- **EST**: `0 9 * * 1-5` (09:00 AM EST)
- **PST**: `0 9 * * 1-5` (09:00 AM PST)

---

## 🧪 Testing de Workflows

### Test Manual desde n8n
1. Abre el workflow en n8n
2. Click en **"Execute Workflow"**
3. Verifica la salida de cada nodo
4. Revisa logs en caso de error

### Test desde CLI
```powershell
# Test endpoints manualmente
Invoke-RestMethod http://localhost:8080/health
Invoke-RestMethod http://localhost:8080/daily_summary?market=IBEX35

# Ver logs de n8n
docker logs n8n -f

# Ver logs de la API
docker logs mcp_finance -f
```

---

## 📊 Monitoreo

### Verificar ejecuciones en n8n
1. Accede a n8n: http://localhost:5678
2. Menú lateral → **"Executions"**
3. Filtra por workflow y fecha
4. Revisa ejecuciones fallidas

### Logs de la API
```powershell
# Ver logs en tiempo real
docker logs mcp_finance -f --tail 100

# Buscar errores
docker logs mcp_finance 2>&1 | Select-String "ERROR"

# Ver últimas predicciones
docker exec db_finanzas psql -U finanzas -d indices -c "SELECT prediction_date, symbol, COUNT(*) FROM ml_predictions GROUP BY prediction_date, symbol ORDER BY prediction_date DESC LIMIT 10;"
```

---

## 🚨 Troubleshooting

### Workflow no se ejecuta
- Verificar que el workflow esté **activado** (toggle ON)
- Revisar horario del cron expression
- Comprobar zona horaria del servidor n8n

### Error "API no responde"
```powershell
# Verificar que el contenedor esté corriendo
docker ps | Select-String mcp_finance

# Probar endpoint manualmente
Invoke-RestMethod http://localhost:8080/health

# Revisar logs
docker logs mcp_finance --tail 50
```

### Error en validación
- Verificar que existan precios para ayer: `SELECT * FROM prices WHERE date = CURRENT_DATE - 1`
- Verificar que existan predicciones: `SELECT * FROM ml_predictions WHERE prediction_date = CURRENT_DATE - 1`

### Email no se envía
- Verificar credenciales SMTP/Gmail en n8n
- Comprobar límites de envío diario
- Revisar carpeta de spam

---

## 📚 Recursos Adicionales

- **n8n Docs**: https://docs.n8n.io/
- **Cron Expression Generator**: https://crontab.guru/
- **Google Gemini API**: https://ai.google.dev/docs
- **FastAPI Docs (API)**: http://localhost:8080/docs

---

## 🎯 Próximos Pasos

1. **Importar** los 3 workflows a n8n
2. **Configurar** credenciales para Gemini y Email (workflow 3)
3. **Activar** workflows 1 y 2
4. **Test manual** de cada workflow
5. **Monitorear** ejecuciones durante 1 semana
6. **Ajustar** horarios según necesidades

---

## 📝 Notas

- Los workflows están diseñados para ejecutarse de **lunes a viernes** (días laborables)
- Los 3 índices se procesan **secuencialmente** para evitar sobrecarga
- La validación solo funciona si hay **precios disponibles** para el día anterior
- El reentrenamiento usa **todos los datos históricos** disponibles en BD
- Los modelos antiguos se **limpian automáticamente** (se mantienen últimos 7 días)

---

**Versión**: 1.0  
**Fecha**: Diciembre 2025  
**Mercados**: IBEX35, SP500, NIKKEI  
**Modelos ML**: 7 (LinearRegression, RandomForest, Prophet, XGBoost, SVR, LightGBM, CatBoost)
