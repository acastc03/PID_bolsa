#!/bin/bash
# Script de instalación automática de la integración con Claude Desktop

set -e  # Salir si hay error

echo "🤖 Instalación de integración con Claude Desktop"
echo "================================================"
echo ""

# Detectar el directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "📁 Directorio del proyecto: $PROJECT_DIR"
echo ""

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip3 install -r "$PROJECT_DIR/mcp_server_claude/requirements.txt"
echo "✅ Dependencias instaladas"
echo ""

# Detectar el sistema operativo
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CONFIG_DIR="$HOME/Library/Application Support/Claude"
    CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    CONFIG_DIR="$APPDATA/Claude"
    CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"
else
    # Linux
    CONFIG_DIR="$HOME/.config/Claude"
    CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"
fi

echo "🔍 Configuración de Claude Desktop: $CONFIG_FILE"
echo ""

# Crear directorio si no existe
mkdir -p "$CONFIG_DIR"

# Leer credenciales de .env
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "📄 Leyendo configuración desde .env..."
    source "$PROJECT_DIR/.env"
else
    echo "⚠️  Archivo .env no encontrado, usando valores por defecto"
    POSTGRES_PORT=15433
    POSTGRES_USER=finanzas
    POSTGRES_PASSWORD=finanzas_pass
    POSTGRES_DB=indices
fi

# Crear configuración JSON
echo "📝 Creando configuración de Claude Desktop..."

cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "finance-predictor": {
      "command": "python3",
      "args": [
        "$PROJECT_DIR/mcp_server_claude/server.py"
      ],
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "$POSTGRES_PORT",
        "DB_NAME": "$POSTGRES_DB",
        "DB_USER": "$POSTGRES_USER",
        "DB_PASS": "$POSTGRES_PASSWORD",
        "PYTHONPATH": "$PROJECT_DIR"
      }
    }
  }
}
EOF

echo "✅ Configuración creada en: $CONFIG_FILE"
echo ""

# Verificar que PostgreSQL está corriendo
echo "🔍 Verificando conexión a PostgreSQL..."
if pg_isready -h localhost -p "$POSTGRES_PORT" > /dev/null 2>&1; then
    echo "✅ PostgreSQL está corriendo en puerto $POSTGRES_PORT"
else
    echo "⚠️  PostgreSQL no está corriendo. Ejecuta: docker-compose up -d db"
fi
echo ""

# Mostrar la configuración
echo "📋 Resumen de la configuración:"
echo "   • Servidor MCP: $PROJECT_DIR/mcp_server_claude/server.py"
echo "   • Base de datos: localhost:$POSTGRES_PORT"
echo "   • Database: $POSTGRES_DB"
echo "   • Usuario: $POSTGRES_USER"
echo ""

echo "🎉 ¡Instalación completada!"
echo ""
echo "📌 Próximos pasos:"
echo "   1. Cierra Claude Desktop completamente"
echo "   2. Vuelve a abrir Claude Desktop"
echo "   3. El servidor MCP se iniciará automáticamente"
echo "   4. Prueba preguntando: '¿Cuál es el precio del IBEX35?'"
echo ""
echo "🔍 Si hay problemas, revisa los logs en:"
echo "   $HOME/Library/Logs/Claude/ (macOS)"
echo ""
echo "📖 Documentación completa: $PROJECT_DIR/mcp_server_claude/README.md"
