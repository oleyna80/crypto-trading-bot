"""
Тестирование обработки ошибок в компонентах Grid Trading Bot.
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

def test_grid_strategy_errors():
    """Проверка обработки некорректных параметров GridStrategy."""
    from models.grid_strategy import GridStrategy
    try:
        # Отрицательное количество уровней
        strategy = GridStrategy(upper_bound=50000, lower_bound=40000, num_levels=-5, amount_per_level=0.001, deposit=1000)
        print("[FAIL] GridStrategy принял отрицательное количество уровней")
        return False
    except ValueError as e:
        print(f"[OK] GridStrategy отклонил отрицательное количество уровней: {e}")
    
    try:
        # Нижняя граница выше верхней
        strategy = GridStrategy(upper_bound=40000, lower_bound=50000, num_levels=10, amount_per_level=0.001, deposit=1000)
        print("[FAIL] GridStrategy принял некорректные границы")
        return False
    except ValueError as e:
        print(f"[OK] GridStrategy отклонил некорректные границы: {e}")
    
    try:
        # Нулевой депозит
        strategy = GridStrategy(upper_bound=50000, lower_bound=40000, num_levels=10, amount_per_level=0.001, deposit=0)
        print("[FAIL] GridStrategy принял нулевой депозит")
        return False
    except ValueError as e:
        print(f"[OK] GridStrategy отклонил нулевой депозит: {e}")
    
    return True

def test_exchange_api_errors():
    """Проверка обработки ошибок сети в ExchangeAPI."""
    from models.exchange_api import ExchangeAPI
    with patch('models.exchange_api.ccxt.bybit') as mock_bybit:
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("Network error")
        mock_bybit.return_value = mock_exchange
        
        api = ExchangeAPI(use_testnet=True)
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 2)
        try:
            data = api.fetch_historical_data('BTC/USDT', '15m', start, end)
            # Метод должен вернуть пустой DataFrame при ошибке
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 0
            print("[OK] ExchangeAPI вернул пустой DataFrame при ошибке сети")
            return True
        except Exception as e:
            print(f"[FAIL] ExchangeAPI выбросил исключение вместо обработки: {e}")
            return False

def test_backtester_empty_data():
    """Проверка обработки пустых данных в Backtester."""
    from models.grid_strategy import GridStrategy
    from services.backtester import Backtester
    
    # Пустой DataFrame
    empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    strategy = GridStrategy(upper_bound=50000, lower_bound=40000, num_levels=10, amount_per_level=0.001, deposit=1000)
    backtester = Backtester(initial_balance=1000)
    try:
        metrics = backtester.run_backtest(empty_data, strategy)
        # Ожидаем, что метрики будут пустыми (нет сделок)
        assert metrics == {}
        print("[OK] Backtester обработал пустые данные без ошибок, метрики пусты")
        return True
    except Exception as e:
        print(f"[FAIL] Backtester упал на пустых данных: {e}")
        return False

def test_data_loader_errors():
    """Проверка обработки ошибок в DataLoader."""
    from services.data_loader import DataLoader
    with patch('services.data_loader.ExchangeAPI') as MockExchangeAPI:
        mock_api = MagicMock()
        mock_api.fetch_historical_data.side_effect = Exception("API недоступен")
        MockExchangeAPI.return_value = mock_api
        
        loader = DataLoader(cache_dir='./test_error_cache')
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 2)
        try:
            df = loader.load_historical_data('INVALID/SYMBOL', '15m', start, end, use_cache=False)
            # Метод должен вернуть пустой DataFrame
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            print("[OK] DataLoader вернул пустой DataFrame при ошибке API")
            return True
        except Exception as e:
            print(f"[FAIL] DataLoader выбросил исключение: {e}")
            return False

def run_all_error_tests():
    """Запуск всех тестов обработки ошибок."""
    print("=== Тестирование обработки ошибок ===")
    results = []
    results.append(test_grid_strategy_errors())
    results.append(test_exchange_api_errors())
    results.append(test_backtester_empty_data())
    results.append(test_data_loader_errors())
    
    passed = sum(results)
    total = len(results)
    print(f"\nИтог: {passed}/{total} тестов пройдено")
    return all(results)

if __name__ == "__main__":
    success = run_all_error_tests()
    sys.exit(0 if success else 1)