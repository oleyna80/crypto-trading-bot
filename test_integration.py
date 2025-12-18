"""
Интеграционное тестирование API (с мок-данными).
"""
import sys
sys.path.append('.')

from models.exchange_api import ExchangeAPI
from services.data_loader import DataLoader
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock

logging.basicConfig(level=logging.WARNING)

def test_exchange_api_mock():
    """Тест ExchangeAPI с моком ccxt."""
    print("Тест 1: ExchangeAPI с моком ccxt")
    try:
        # Мокаем ccxt.bybit
        with patch('models.exchange_api.ccxt.bybit') as mock_bybit:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.return_value = [
                [1609459200000, 40000, 41000, 39000, 40500, 1000],
                [1609459260000, 40500, 41500, 39500, 41000, 1100],
            ]
            mock_bybit.return_value = mock_exchange
            
            api = ExchangeAPI(use_testnet=True)
            # Вызываем метод, который использует exchange внутри
            from datetime import datetime
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 2)
            data = api.fetch_historical_data(
                symbol='BTC/USDT',
                timeframe='15m',
                start_date=start,
                end_date=end
            )
            # Проверяем, что данные преобразованы в DataFrame
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 2
            assert 'open' in data.columns
            assert 'high' in data.columns
            assert 'low' in data.columns
            assert 'close' in data.columns
            assert 'volume' in data.columns
            print(f"  [OK] Данные загружены, строк: {len(data)}")
            return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def test_data_loader_integration():
    """Тест DataLoader с моком ExchangeAPI."""
    print("Тест 2: DataLoader интеграция")
    try:
        with patch('services.data_loader.ExchangeAPI') as MockExchangeAPI:
            mock_api = MagicMock()
            mock_api.fetch_historical_data.return_value = pd.DataFrame({
                'open': [40000, 40500],
                'high': [41000, 41500],
                'low': [39000, 39500],
                'close': [40500, 41000],
                'volume': [1000, 1100]
            }, index=pd.date_range('2025-01-01', periods=2, freq='15min'))
            MockExchangeAPI.return_value = mock_api
            
            loader = DataLoader()
            from datetime import datetime
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 2)
            data = loader.load_historical_data(
                symbol='BTC/USDT',
                timeframe='15m',
                start_date=start,
                end_date=end,
                use_cache=False
            )
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 2
            # Проверяем, что кэширование работает (вызов метода должен быть один раз)
            assert mock_api.fetch_historical_data.call_count == 1
            print("  [OK] DataLoader загрузил данные через мок API")
            return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def test_backtester_integration():
    """Тест интеграции Backtester с GridStrategy и данными."""
    print("Тест 3: Интеграция Backtester + GridStrategy + данные")
    try:
        # Создаём искусственные данные
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=30, freq='15min'),
            'open': [45000.0] * 30,
            'high': [45100.0] * 30,
            'low': [44000.0] * 30,
            'close': [44500.0] * 30,
            'volume': [100.0] * 30
        })
        # Устанавливаем индекс как timestamp для совместимости
        data.set_index('timestamp', inplace=True)
        
        from models.grid_strategy import GridStrategy
        from services.backtester import Backtester
        
        strategy = GridStrategy(
            upper_bound=50000.0,
            lower_bound=40000.0,
            num_levels=5,
            amount_per_level=0.001,
            deposit=1000.0
        )
        backtester = Backtester(initial_balance=1000.0)
        metrics = backtester.run_backtest(data, strategy)
        
        # Проверяем, что метрики рассчитаны
        assert 'total_pnl' in metrics
        assert 'total_trades' in metrics
        # Проверяем, что equity_curve заполнена
        assert len(backtester.equity_curve) == len(data)
        print(f"  [OK] Интеграционный тест пройден, trades: {metrics['total_trades']}")
        return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def test_error_handling():
    """Тест обработки ошибок в API."""
    print("Тест 4: Обработка ошибок API")
    try:
        with patch('models.exchange_api.ccxt.bybit') as mock_bybit:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.side_effect = Exception("Network error")
            mock_bybit.return_value = mock_exchange
            
            api = ExchangeAPI(use_testnet=True)
            from datetime import datetime
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 2)
            # Должен вернуть пустой DataFrame при ошибке
            data = api.fetch_historical_data(
                symbol='BTC/USDT',
                timeframe='15m',
                start_date=start,
                end_date=end
            )
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 0
            print("  [OK] Ошибка обработана, возвращён пустой DataFrame")
            return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def test_cache_integration():
    """Тест кэширования данных."""
    print("Тест 5: Кэширование данных")
    try:
        import os
        import pickle
        from services.data_loader import DataLoader
        
        # Удаляем старый кэш, если есть
        cache_dir = './test_cache'
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
        
        with patch('services.data_loader.ExchangeAPI') as MockExchangeAPI:
            mock_api = MagicMock()
            mock_api.fetch_historical_data.return_value = pd.DataFrame({
                'open': [40000],
                'high': [41000],
                'low': [39000],
                'close': [40500],
                'volume': [1000]
            }, index=pd.date_range('2025-01-01', periods=1, freq='15min'))
            MockExchangeAPI.return_value = mock_api
            
            loader = DataLoader(cache_dir=cache_dir)
            from datetime import datetime
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 2)
            # Первый вызов — загрузка через API
            data1 = loader.load_historical_data(
                symbol='BTC/USDT',
                timeframe='15m',
                start_date=start,
                end_date=end,
                use_cache=True
            )
            assert mock_api.fetch_historical_data.call_count == 1
            # Проверим, что кэш-файл создан
            cache_file = os.path.join(cache_dir, 'BTC_USDT_15m_20250101_20250102.pkl')
            assert os.path.exists(cache_file), f"Кэш-файл не создан: {cache_file}"
            # Второй вызов — должен взять из кэша
            data2 = loader.load_historical_data(
                symbol='BTC/USDT',
                timeframe='15m',
                start_date=start,
                end_date=end,
                use_cache=True
            )
            # Кэширование должно предотвратить второй вызов API
            # Проверим, что данные идентичны
            assert len(data1) == len(data2)
            # Проверим, что второй вызов не вызвал fetch_historical_data (кэш сработал)
            assert mock_api.fetch_historical_data.call_count == 1, "Второй вызов API не должен был произойти"
            print("  [OK] Кэширование работает (данные загружены из кэша)")
            return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Запуск всех тестов."""
    print("=== Интеграционное тестирование API ===")
    results = []
    results.append(test_exchange_api_mock())
    results.append(test_data_loader_integration())
    results.append(test_backtester_integration())
    results.append(test_error_handling())
    results.append(test_cache_integration())
    
    passed = sum(results)
    total = len(results)
    print(f"\nИтог: {passed}/{total} тестов пройдено")
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)