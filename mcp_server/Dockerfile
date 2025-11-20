# -----------------------------
# Imagen base
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# Variables de entorno
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# Instalar dependencias del sistema necesarias solo una vez
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        build-essential \
        git \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Crear directorio de trabajo
# -----------------------------
WORKDIR /app

# -----------------------------
# Copiar y instalar dependencias de Python
# -----------------------------
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copiar código fuente
# -----------------------------
COPY . .

# -----------------------------
# Limpiar build-essential si no se necesita más
# -----------------------------
RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------
# Comando por defecto
# -----------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]