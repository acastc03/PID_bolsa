import sys
sys.path.insert(0, '/app')
from scripts.config import get_db_conn

conn = get_db_conn()
cur = conn.cursor()

print("\n=== RESUMEN DE PREDICCIONES GUARDADAS ===\n")

# Predicciones por símbolo
cur.execute("""
    SELECT symbol, 
           COUNT(DISTINCT prediction_date) as dias, 
           COUNT(*) as total_predicciones 
    FROM ml_predictions 
    GROUP BY symbol 
    ORDER BY symbol
""")

print("Predicciones por mercado:")
for row in cur.fetchall():
    print(f"  {row['symbol']}: {row['dias']} días distintos, {row['total_predicciones']} predicciones totales")

# Rango de fechas
cur.execute("""
    SELECT 
        MIN(prediction_date) as primera_fecha,
        MAX(prediction_date) as ultima_fecha
    FROM ml_predictions
""")
row = cur.fetchone()
print(f"\nRango de fechas: {row['primera_fecha']} a {row['ultima_fecha']}")

# Modelos únicos
cur.execute("""
    SELECT DISTINCT model_name 
    FROM ml_predictions 
    ORDER BY model_name
""")
modelos = [r['model_name'] for r in cur.fetchall()]
print(f"\nModelos entrenados: {', '.join(modelos)}")

conn.close()
print("\n✅ Verificación completada\n")
