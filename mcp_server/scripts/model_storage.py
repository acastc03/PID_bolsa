# mcp_server/scripts/model_storage.py

import os
import pickle
import joblib
from datetime import datetime
from pathlib import Path
from . import logger

# Directorio donde se guardan los modelos
MODELS_DIR = Path("/app/data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_path(symbol: str, model_name: str, date: str = None) -> Path:
    """
    Genera la ruta del archivo del modelo.
    
    Args:
        symbol: Símbolo del activo (ej: ^IBEX)
        model_name: Nombre del modelo (ej: LinearRegression, XGBoost)
        date: Fecha de entrenamiento (formato YYYY-MM-DD). Si es None, usa 'latest'
    
    Returns:
        Path al archivo del modelo
    """
    if date is None:
        date = "latest"
    
    # Sanitizar el símbolo para nombre de archivo
    safe_symbol = symbol.replace("^", "").replace("/", "_")
    filename = f"{safe_symbol}_{model_name}_{date}.pkl"
    return MODELS_DIR / filename


def save_model(model, symbol: str, model_name: str, date: str = None, metadata: dict = None):
    """
    Guarda un modelo entrenado en disco.
    
    Args:
        model: Modelo entrenado
        symbol: Símbolo del activo
        model_name: Nombre del modelo
        date: Fecha de entrenamiento (YYYY-MM-DD)
        metadata: Información adicional (métricas, features usadas, etc.)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    model_path = get_model_path(symbol, model_name, date)
    
    # Guardar modelo y metadata juntos
    model_data = {
        "model": model,
        "symbol": symbol,
        "model_name": model_name,
        "training_date": date,
        "saved_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        # También guardar como "latest" para acceso rápido
        latest_path = get_model_path(symbol, model_name, "latest")
        with open(latest_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Modelo guardado: {model_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error guardando modelo {model_name}: {e}")
        return False


def load_model(symbol: str, model_name: str, date: str = None):
    """
    Carga un modelo desde disco.
    
    Args:
        symbol: Símbolo del activo
        model_name: Nombre del modelo
        date: Fecha específica (YYYY-MM-DD) o None para cargar el último
    
    Returns:
        Diccionario con modelo y metadata, o None si no existe
    """
    if date is None:
        date = "latest"
    
    model_path = get_model_path(symbol, model_name, date)
    
    if not model_path.exists():
        logger.warning(f"⚠️ Modelo no encontrado: {model_path}")
        return None
    
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        logger.info(f"✅ Modelo cargado: {model_path}")
        return model_data
    except Exception as e:
        logger.error(f"❌ Error cargando modelo {model_name}: {e}")
        return None


def model_exists(symbol: str, model_name: str, date: str = None) -> bool:
    """Verifica si existe un modelo guardado."""
    if date is None:
        date = "latest"
    model_path = get_model_path(symbol, model_name, date)
    return model_path.exists()


def delete_old_models(symbol: str, keep_latest: int = 5):
    """
    Elimina modelos antiguos, conservando solo los últimos N días.
    
    Args:
        symbol: Símbolo del activo
        keep_latest: Número de versiones diarias a conservar
    """
    safe_symbol = symbol.replace("^", "").replace("/", "_")
    pattern = f"{safe_symbol}_*_20*.pkl"  # Solo archivos con fecha
    
    model_files = sorted(MODELS_DIR.glob(pattern), reverse=True)
    
    # Agrupar por nombre de modelo
    models_by_name = {}
    for file in model_files:
        parts = file.stem.split("_")
        model_name = "_".join(parts[1:-1])  # Excluir símbolo y fecha
        
        if model_name not in models_by_name:
            models_by_name[model_name] = []
        models_by_name[model_name].append(file)
    
    # Eliminar versiones antiguas de cada modelo
    deleted_count = 0
    for model_name, files in models_by_name.items():
        if len(files) > keep_latest:
            for old_file in files[keep_latest:]:
                try:
                    old_file.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ Eliminado modelo antiguo: {old_file.name}")
                except Exception as e:
                    logger.error(f"Error eliminando {old_file}: {e}")
    
    logger.info(f"✅ Limpieza completada. {deleted_count} modelos eliminados.")
    return deleted_count


def get_model_info(symbol: str) -> dict:
    """
    Obtiene información sobre los modelos disponibles para un símbolo.
    
    Returns:
        Dict con lista de modelos y sus fechas
    """
    safe_symbol = symbol.replace("^", "").replace("/", "_")
    pattern = f"{safe_symbol}_*.pkl"
    
    model_files = MODELS_DIR.glob(pattern)
    
    models_info = []
    for file in model_files:
        try:
            with open(file, "rb") as f:
                model_data = pickle.load(f)
            
            models_info.append({
                "file": file.name,
                "model_name": model_data.get("model_name"),
                "training_date": model_data.get("training_date"),
                "saved_at": model_data.get("saved_at"),
                "metadata": model_data.get("metadata", {})
            })
        except Exception as e:
            logger.error(f"Error leyendo info de {file}: {e}")
    
    return {
        "symbol": symbol,
        "models": sorted(models_info, key=lambda x: x["saved_at"], reverse=True)
    }
