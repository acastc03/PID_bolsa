@echo off
REM ============================================
REM MCP Finance Server - Simple Docker
REM Windows Batch Script
REM ============================================

REM Get the directory where the script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    exit /b 1
)

REM Run the container (builds image if needed)
echo [INFO] Starting MCP Finance Server (building if needed)...
docker run --rm ^
    --network host ^
    -v "%PROJECT_ROOT%\data\models:/app/data/models" ^
    -e DB_HOST=localhost ^
    -e DB_PORT=15433 ^
    -e DB_NAME=indices ^
    -e DB_USER=finanzas ^
    -e DB_PASS=finanzas_pass ^
    %SCRIPT_DIR%
