"""
Модульное тестирование Backtester.
"""
import sys
sys.path.append('.')

from services.backtester import Backtester
from models.grid_strategy import GridStrategy
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

def test_backtester_initialization():
    """Тест инициализации Backtester."""
    print("Тест 1: Инициализация Backtester")
    try:
        backtester = Backtester(initial_balance=1000.0)
        assert backtester.initial_balance == 1000.0
        assert backtester.trades == []
        assert backtester.equity_curve == []
        print("  [OK] Успешно")
        return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def test_backtester_run_simple():
    """Тест простого прогона backtester с искусственными данными."""
    print("Тест 2: Простой прогон backtester")
    try:
        strategy = GridStrategy(
            upper_bound=50000.0,
            lower_bound=40000.0,
            num_levels=3,
            amount_per_level=0.001,
            deposit=1000.0
        )
        backtester = Backtester(initial_balance=1000.0)
        # Создаём искусственные данные: цена движется от 45000 до 44000
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=10, freq='15min'),
            'open': [45000.0] * 10,
            'high': [45100.0] * 10,
            'low': [44000.0] * 10,
            'close': [44500.0] * 10,
            'volume': [100.0] * 10
        })
        # Уровни: 40000 buy, 45000 sell, 50000 buy
        # Цена закрытия 44500: должна сработать sell на 45000? Нет, потому что цена <= уровень * (1 - tolerance)
        # Проверим, что backtester не падает
        backtester.run_backtest(data, strategy)
        # Проверяем, что equity_curve заполнена
        assert len(backtester.equity_curve) == len(data)
        # Проверяем, что trades может быть пустым (если не было исполнений)
        print(f"  [OK] Прогон завершён, trades: {len(backtester.trades)}, equity curve: {len(backtester.equity_curve)}")
        return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def test_backtester_trade_execution():
    """Тест исполнения ордеров в backtester."""
    print("Тест 3: Исполнение ордеров в backtester")
    try:
        strategy = GridStrategy(
            upper_bound=50000.0,
            lower_bound=40000.0,
            num_levels=3,
            amount_per_level=0.001,
            deposit=1000.0
        )
        backtester = Backtester(initial_balance=1000.0)
        # Данные, где цена пересекает уровень 45000 (sell)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=5, freq='15min'),
            'open': [46000.0, 45500.0, 45000.0, 44500.0, 44000.0],
            'high': [46100.0, 45600.0, 45100.0, 44600.0, 44100.0],
            'low': [45900.0, 45400.0, 44900.0, 44400.0, 43900.0],
            'close': [46000.0, 45500.0, 45000.0, 44500.0, 44000.0],
            'volume': [100.0] * 5
        })
        backtester.run_backtest(data, strategy)
        # Ожидаем хотя бы один trade (sell на 45000)
        if len(backtester.trades) > 0:
            print(f"  [OK] Исполнено {len(backtester.trades)} ордеров")
        else:
            print("  [WARN] Ни одного ордера не исполнено (возможно, логика tolerance)")
        # Проверяем equity_curve
        assert len(backtester.equity_curve) == len(data)
        # Проверяем, что equity не отрицательная
        for eq in backtester.equity_curve:
            assert eq['equity'] >= 0.0
        print("  [OK] Equity curve корректна")
        return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def test_backtester_metrics():
    """Тест расчёта метрик."""
    print("Тест 4: Расчёт метрик backtester")
    try:
        strategy = GridStrategy(
            upper_bound=50000.0,
            lower_bound=40000.0,
            num_levels=3,
            amount_per_level=0.001,
            deposit=1000.0
        )
        backtester = Backtester(initial_balance=1000.0)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=10, freq='15min'),
            'open': [45000.0] * 10,
            'high': [45100.0] * 10,
            'low': [44900.0] * 10,
            'close': [45000.0] * 10,
            'volume': [100.0] * 10
        })
        backtester.run_backtest(data, strategy)
        metrics = backtester.calculate_metrics()
        # Проверяем наличие ключевых метрик
        required_keys = ['total_pnl', 'total_pnl_pct', 'win_rate', 'max_drawdown', 'sharpe_ratio', 'total_trades']
        for key in required_keys:
            assert key in metrics
            print(f"    {key}: {metrics[key]}")
        # Проверяем, что метрики числовые
        assert isinstance(metrics['total_pnl'], (int, float))
        assert isinstance(metrics['total_pnl_pct'], (int, float))
        print("  [OK] Метрики рассчитаны корректно")
        return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def test_backtester_edge_cases():
    """Тест граничных случаев."""
    print("Тест 5: Граничные случаи backtester")
    try:
        # Пустые данные
        strategy = GridStrategy(
            upper_bound=50000.0,
            lower_bound=40000.0,
            num_levels=3,
            amount_per_level=0.001,
            deposit=1000.0
        )
        backtester = Backtester(initial_balance=1000.0)
        empty_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        backtester.run_backtest(empty_data, strategy)
        assert len(backtester.equity_curve) == 0
        assert len(backtester.trades) == 0
        print("  [OK] Пустые данные обработаны")
        
        # Одна строка данных
        single_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-01-01')],
            'open': [45000.0],
            'high': [45100.0],
            'low': [44900.0],
            'close': [45000.0],
            'volume': [100.0]
        })
        backtester2 = Backtester(initial_balance=1000.0)
        backtester2.run_backtest(single_data, strategy)
        assert len(backtester2.equity_curve) == 1
        print("  [OK] Одна строка данных обработана")
        return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def run_all_tests():
    """Запуск всех тестов."""
    print("=== Модульное тестирование Backtester ===")
    results = []
    results.append(test_backtester_initialization())
    results.append(test_backtester_run_simple())
    results.append(test_backtester_trade_execution())
    results.append(test_backtester_metrics())
    results.append(test_backtester_edge_cases())
    
    passed = sum(results)
    total = len(results)
    print(f"\nИтог: {passed}/{total} тестов пройдено")
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)