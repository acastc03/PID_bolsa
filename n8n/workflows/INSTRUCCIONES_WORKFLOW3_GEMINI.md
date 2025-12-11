# 📧 Completar Workflow 3 - Email Report con Gemini

## 🎯 Estado Actual
El workflow tiene:
- ✅ Schedule Evening 8:00 PM
- ✅ Loop Markets (genera 3 items: IBEX35, SP500, NIKKEI)
- ✅ Get Daily Summary (obtiene datos de cada mercado)
- ✅ Prepare Data

## 🔧 Nodos a Agregar Manualmente en n8n

### 1️⃣ Nodo AGGREGATE (Combinar datos de 3 mercados)

**Después de**: "Prepare Data"

**Tipo**: `Aggregate`
**Configuración**:
```
Operation: Aggregate All Items Into One
Options:
  - Keep Input Data: true
```

Este nodo recibe los 3 items (uno por mercado) y los combina en un solo array.

---

### 2️⃣ Nodo CODE (Preparar prompt para Gemini)

**Después de**: Aggregate

**Tipo**: `Code`
**Configuración**:
```javascript
// Extraer datos de los 3 mercados
const markets = $input.all();

// Construir resumen estructurado
let summary = "Genera un reporte profesional de mercados financieros para hoy.\n\n";

markets.forEach(item => {
  const data = JSON.parse(item.json.market_data);
  
  summary += `\n## ${data.market}\n`;
  summary += `- Precio actual: ${data.price?.current || 'N/A'}\n`;
  summary += `- Cambio: ${data.price?.change || 'N/A'} (${data.price?.change_percent || 'N/A'}%)\n`;
  summary += `- RSI: ${data.indicators?.rsi || 'N/A'}\n`;
  summary += `- Señal: ${data.signal || 'N/A'}\n`;
  
  if (data.ml_performance) {
    summary += `- Predicción ensemble: ${data.ml_performance.ensemble_signal || 'N/A'}\n`;
    summary += `- Modelos activos: ${data.ml_performance.models_count || 0}\n`;
  }
  
  if (data.news && data.news.length > 0) {
    summary += `- Últimas noticias:\n`;
    data.news.slice(0, 3).forEach(n => {
      summary += `  * ${n.title}\n`;
    });
  }
  
  summary += '\n';
});

summary += "\nFormato del reporte:\n";
summary += "1. Resumen ejecutivo (2-3 líneas)\n";
summary += "2. Análisis por mercado (IBEX35, SP500, NIKKEI)\n";
summary += "3. Señales de trading y recomendaciones\n";
summary += "4. Conclusión\n\n";
summary += "Usa un tono profesional pero accesible. Incluye emojis relevantes.";

return [{
  json: {
    prompt: summary
  }
}];
```

---

### 3️⃣ Nodo HTTP REQUEST (Llamada a Gemini API)

**Después de**: Code

**Tipo**: `HTTP Request`
**Configuración**:
```
Method: POST
URL: https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent

Authentication: None (la API key va en la URL o Header)

Query Parameters:
  key: TU_API_KEY_DE_GEMINI

Headers:
  Content-Type: application/json

Body (JSON):
{
  "contents": [{
    "parts": [{
      "text": "={{ $json.prompt }}"
    }]
  }],
  "generationConfig": {
    "temperature": 0.7,
    "maxOutputTokens": 2048
  }
}

Options:
  - Response Format: JSON
```

**🔑 Obtener API Key**:
1. Ir a https://aistudio.google.com/
2. Click en "Get API Key"
3. Crear nueva key o usar existente
4. Copiar la key

**Alternativa con credenciales en n8n**:
- En vez de poner la key en la URL, puedes:
  - Click en "Add option" → "Add Header"
  - Name: `x-goog-api-key`
  - Value: `{{ $credentials.geminiApiKey }}`
  - Y configurar las credenciales en n8n Settings

---

### 4️⃣ Nodo CODE (Extraer respuesta de Gemini)

**Después de**: HTTP Request (Gemini)

**Tipo**: `Code`
**Configuración**:
```javascript
// Gemini devuelve la respuesta en un formato específico
const response = $input.first().json;

let reportText = '';

try {
  // Extraer el texto generado
  reportText = response.candidates[0].content.parts[0].text;
} catch (error) {
  reportText = 'Error al generar el reporte: ' + error.message;
}

return [{
  json: {
    report: reportText,
    timestamp: new Date().toISOString(),
    generated_by: 'Gemini 1.5 Flash'
  }
}];
```

---

### 5️⃣ Nodo GMAIL / SMTP (Enviar Email)

**Después de**: Code (Extraer respuesta)

**Opción A: Gmail OAuth2**

**Tipo**: `Gmail`
**Configuración**:
```
Operation: Send Email

To: tu-email@example.com
Subject: 📈 Reporte Diario Mercados - {{ $now.format('DD/MM/YYYY') }}
Email Type: Text or HTML

Message (HTML):
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
    .content { padding: 20px; white-space: pre-wrap; }
    .footer { background: #f4f4f4; padding: 10px; text-align: center; font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <div class="header">
    <h1>📊 Reporte Diario de Mercados</h1>
    <p>{{ $now.format('dddd, DD MMMM YYYY') }}</p>
  </div>
  <div class="content">
    {{ $json.report }}
  </div>
  <div class="footer">
    <p>Generado automáticamente por PID Bolsa | Powered by Gemini AI</p>
    <p>IBEX35 🇪🇸 | S&P500 🇺🇸 | NIKKEI 🇯🇵</p>
  </div>
</body>
</html>
```

**Configurar credenciales Gmail**:
1. En n8n: Settings → Credentials → Add Credential
2. Buscar "Gmail OAuth2"
3. Seguir el wizard para conectar tu cuenta Google

---

**Opción B: SMTP Genérico**

**Tipo**: `Send Email`
**Configuración**:
```
From Email: tu-email@gmail.com
To Email: destinatario@example.com
Subject: 📈 Reporte Diario Mercados - {{ $now.format('DD/MM/YYYY') }}

Message: (mismo HTML de arriba)

SMTP Connection:
  Host: smtp.gmail.com
  Port: 587
  SSL/TLS: Use TLS
  User: tu-email@gmail.com
  Password: contraseña-de-aplicación-de-google
```

**⚠️ Para Gmail SMTP necesitas**:
1. Activar "Verificación en 2 pasos" en tu cuenta Google
2. Generar una "Contraseña de aplicación":
   - https://myaccount.google.com/apppasswords
   - Seleccionar "Correo" y "Otro (nombre personalizado)"
   - Copiar la contraseña de 16 caracteres

---

## 🎨 Diagrama Completo del Workflow 3

```
Schedule 8PM
    ↓
Loop Markets (Code)
    ↓
Get Daily Summary (HTTP) ← procesa 3 items en paralelo
    ↓
Prepare Data (Set)
    ↓
[AGREGAR] Aggregate ← combina 3 items en 1
    ↓
[AGREGAR] Code (Preparar prompt)
    ↓
[AGREGAR] HTTP Request (Gemini API)
    ↓
[AGREGAR] Code (Extraer respuesta)
    ↓
[AGREGAR] Gmail / SMTP
    ↓
FIN ✅
```

---

## 🧪 Testing

1. **Test del nodo Gemini solo**:
   - Ejecuta el workflow hasta el nodo Code (Preparar prompt)
   - Copia el prompt generado
   - Pruébalo manualmente en https://aistudio.google.com/

2. **Test del email**:
   - Primero prueba con un texto fijo en vez de Gemini
   - Una vez funcione el email, conecta Gemini

3. **Test completo**:
   - Ejecuta el workflow completo
   - Verifica el email recibido
   - Ajusta el prompt si es necesario

---

## 💡 Tips

- **Límites de Gemini Free**: 15 requests/min, 1500 requests/día
- **Alternativas a Gemini**: OpenAI GPT-4, Claude, Mistral
- **Mejorar el prompt**: Agrega ejemplos del formato deseado
- **HTML del email**: Puedes usar un template más elaborado
- **Adjuntar gráficos**: Usa Chart.js o similar antes del email

---

## 🔧 Troubleshooting

**Error "API key not valid"**:
- Verifica la key en https://aistudio.google.com/
- Asegúrate de que la API está habilitada

**Email no llega**:
- Revisa carpeta de spam
- Verifica credenciales SMTP/Gmail
- Comprueba límites de envío de Gmail (500/día)

**Gemini devuelve texto vacío**:
- El prompt puede ser demasiado largo
- Reduce la cantidad de noticias incluidas
- Aumenta `maxOutputTokens`

**Timeout en Gemini**:
- Gemini puede tardar 5-10 segundos
- En n8n: Settings del nodo HTTP → Timeout: 30000ms

---

## 📚 Recursos

- **Gemini API Docs**: https://ai.google.dev/docs
- **n8n Gmail Node**: https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-base.gmail/
- **HTML Email Templates**: https://www.htmlemailcheck.com/
- **Prompt Engineering**: https://ai.google.dev/gemini-api/docs/prompting-intro

---

**¿Necesitas ayuda?**
- Gemini no responde bien → ajusta el prompt
- Email no se envía → revisa credenciales
- Formato HTML roto → valida en https://validator.w3.org/
