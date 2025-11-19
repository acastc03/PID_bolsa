"""
Paquete principal de la API.

Re-exporta la instancia de FastAPI definida en app.main
para que pueda importarse simplemente con `from app import app`.
"""

from .main import app  # noqa: F401