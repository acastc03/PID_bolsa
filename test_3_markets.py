import sys
sys.path.insert(0, '/app')
from datetime import date
from scripts.backfill_predictions import backfill_predictions_for_symbol

print('🚀 BACKFILL - 3 MERCADOS GLOBALES')
print('=' * 70)
print('Test: 1 día (2024-12-02)')
print()

markets = [
    ('^IBEX', '🇪🇸 IBEX35 - España'),
    ('^GSPC', '🇺🇸 SP500 - USA'),
    ('^N225', '🇯🇵 NIKKEI - Japón')
]

results = {}
test_date = date(2024, 12, 2)

for symbol, name in markets:
    print(f'📊 {name}')
    
    try:
        backfill_predictions_for_symbol(
            symbol=symbol,
            start_date=test_date,
            end_date=test_date
        )
        results[name] = 'SUCCESS'
        print(f'✅ Completado\n')
    except Exception as e:
        error_msg = str(e)[:60]
        results[name] = f'ERROR: {error_msg}'
        print(f'❌ Error: {error_msg}\n')

print()
print('╔══════════════════════════════════════════════════════════════════╗')
print('║                        📋 RESUMEN                                ║')
print('╠══════════════════════════════════════════════════════════════════╣')
for name, status in results.items():
    icon = '✅' if status == 'SUCCESS' else '❌'
    print(f'║  {icon} {name:40} {status:16} ║')
print('╚══════════════════════════════════════════════════════════════════╝')
