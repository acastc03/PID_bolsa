#!/usr/bin/env python3
"""
Test para verificar que el fix de backfill funciona correctamente.

Ejecutar DENTRO del contenedor:
    docker exec -it mcp_finance python test_backfill_fix.py
"""

import sys
sys.path.insert(0, '/app')

from datetime import date
from scripts.models import _load_features, predict_ensemble

def test_load_features_with_as_of_date():
    """Verifica que _load_features filtra correctamente por fecha."""
    print("🧪 Test 1: _load_features con as_of_date")
    print("-" * 60)
    
    symbol = "^IBEX"
    test_date = date(2024, 12, 1)
    
    # Cargar todos los datos
    df_all = _load_features(symbol)
    print(f"✓ Sin filtro: {len(df_all)} filas")
    print(f"  Rango: {df_all.index[0].date()} a {df_all.index[-1].date()}")
    
    # Cargar solo hasta test_date
    df_limited = _load_features(symbol, as_of_date=test_date)
    print(f"✓ Hasta {test_date}: {len(df_limited)} filas")
    print(f"  Rango: {df_limited.index[0].date()} a {df_limited.index[-1].date()}")
    
    # Verificaciones
    assert len(df_limited) < len(df_all), "❌ df_limited debe tener menos filas"
    assert df_limited.index[-1].date() <= test_date, f"❌ Última fecha debe ser <= {test_date}"
    
    print("✅ Test 1 PASADO: _load_features filtra correctamente\n")
    return True


def test_predict_ensemble_with_as_of_date():
    """Verifica que predict_ensemble respeta as_of_date."""
    print("🧪 Test 2: predict_ensemble con as_of_date")
    print("-" * 60)
    
    symbol = "^IBEX"
    test_date = date(2024, 11, 15)
    
    print(f"Ejecutando predict_ensemble para {symbol} con as_of_date={test_date}")
    print("(Esto puede tardar varios minutos - reentrenando 7 modelos ML...)")
    
    result = predict_ensemble(symbol, as_of_date=test_date, force_retrain=True)
    
    print(f"✓ Señal ensemble: {result['signal_ensemble']}")
    print(f"✓ Número de modelos ML: {len(result['ml_models'])}")
    print(f"✓ Señales ML: {result.get('ml_signals', [])}")
    
    # Verificaciones básicas
    assert 'signal_ensemble' in result, "❌ Falta signal_ensemble"
    assert len(result['ml_models']) > 0, "❌ No hay modelos ML"
    assert result['signal_ensemble'] in [-1, 0, 1], "❌ Señal inválida"
    
    print("✅ Test 2 PASADO: predict_ensemble funciona con as_of_date\n")
    return True


def test_different_dates_different_results():
    """Verifica que fechas diferentes producen resultados diferentes."""
    print("🧪 Test 3: Diferentes fechas producen diferentes resultados")
    print("-" * 60)
    
    symbol = "^IBEX"
    date1 = date(2024, 11, 1)
    date2 = date(2024, 12, 1)
    
    print(f"Cargando datos para {date1}...")
    df1 = _load_features(symbol, as_of_date=date1)
    
    print(f"Cargando datos para {date2}...")
    df2 = _load_features(symbol, as_of_date=date2)
    
    print(f"✓ Datos hasta {date1}: {len(df1)} filas")
    print(f"✓ Datos hasta {date2}: {len(df2)} filas")
    
    # Verificar que hay diferencia
    assert len(df2) > len(df1), f"❌ df2 ({len(df2)}) debe tener más filas que df1 ({len(df1)})"
    
    diff = len(df2) - len(df1)
    print(f"✓ Diferencia: {diff} días adicionales")
    
    print("✅ Test 3 PASADO: Fechas diferentes = datos diferentes\n")
    return True


def main():
    """Ejecuta todos los tests."""
    print("=" * 60)
    print("🔬 TESTS DE VERIFICACIÓN DEL FIX DE BACKFILL")
    print("=" * 60)
    print()
    
    tests = [
        test_load_features_with_as_of_date,
        test_predict_ensemble_with_as_of_date,
        test_different_dates_different_results,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"❌ TEST FALLIDO: {e}\n")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTADOS: {passed} pasados, {failed} fallidos")
    print("=" * 60)
    
    if failed == 0:
        print("✅ TODOS LOS TESTS PASARON - Fix verificado correctamente")
        return 0
    else:
        print("❌ ALGUNOS TESTS FALLARON - Revisar implementación")
        return 1


if __name__ == "__main__":
    sys.exit(main())
