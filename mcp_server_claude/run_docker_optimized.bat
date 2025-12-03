@echo off
REM ============================================
REM MCP Finance Server - Docker Optimized
REM Windows Batch Script
REM ============================================

REM Get the directory where the script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Docker image name
set IMAGE_NAME=mcp-finance-server:latest

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    exit /b 1
)

REM Check if image exists, if not build it
docker image inspect %IMAGE_NAME% >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Building Docker image %IMAGE_NAME%...
    docker build -t %IMAGE_NAME% "%SCRIPT_DIR%"
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to build Docker image
        exit /b 1
    )
    echo [SUCCESS] Docker image built successfully
)

REM Run the container
echo [INFO] Starting MCP Finance Server...
docker run --rm ^
    --network host ^
    -v "%PROJECT_ROOT%\data\models:/app/data/models" ^
    -e DB_HOST=localhost ^
    -e DB_PORT=15433 ^
    -e DB_NAME=indices ^
    -e DB_USER=finanzas ^
    -e DB_PASS=finanzas_pass ^
    %IMAGE_NAME%
