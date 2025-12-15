"""
Backfill de predicciones para los últimos 14 días.
Ejecutar: docker exec mcp_finance python -m scripts.run_backfill_14days
"""
from datetime import date, timedelta
from scripts.config import get_db_conn
from scripts.models import predict_ensemble
from scripts.save_predictions import save_daily_predictions

def run_backfill():
    symbols = ['^IBEX', '^GSPC', '^N225']
    end_date = date.today()
    start_date = end_date - timedelta(days=14)

    conn = get_db_conn()

    for symbol in symbols:
        print(f'\n=== Procesando {symbol} ===')
        cur = conn.cursor()
        cur.execute('''
            SELECT DISTINCT date FROM prices 
            WHERE symbol = %s AND date BETWEEN %s AND %s
            ORDER BY date
        ''', (symbol, start_date, end_date))
        dates = [row['date'] for row in cur.fetchall()]
        print(f'  Fechas con precios: {len(dates)}')
        
        for d in dates:
            cur.execute('''
                SELECT COUNT(*) as cnt FROM ml_predictions 
                WHERE symbol = %s AND prediction_date = %s
            ''', (symbol, d))
            cnt = cur.fetchone()['cnt']
            
            if cnt == 0:
                print(f'  Generando predicciones para {d}...')
                try:
                    result = predict_ensemble(symbol, as_of_date=d, force_retrain=False)
                    if result and result.get('predictions'):
                        save_daily_predictions(
                            symbol=symbol,
                            run_date=d,
                            target_date=d,
                            predictions=result['predictions'],
                            ensemble_signal=result.get('ensemble_signal', 0)
                        )
                        print(f'    OK: {len(result["predictions"])} modelos')
                except Exception as e:
                    print(f'    Error: {e}')
            else:
                print(f'  {d}: ya tiene {cnt} predicciones')

    conn.close()
    print('\n✅ Backfill completado')

if __name__ == '__main__':
    run_backfill()
