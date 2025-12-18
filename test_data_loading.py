"""
Тестирование исправленного метода fetch_historical_data.
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta
from models.exchange_api import ExchangeAPI
import logging

logging.basicConfig(level=logging.INFO)

def test_fetch_historical_data(days: int, timeframe: str = '15m'):
    """Загружает данные за указанное количество дней и проверяет количество свечей."""
    api = ExchangeAPI(use_testnet=True)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"\n=== Тест за {days} дней, таймфрейм {timeframe} ===")
    print(f"Период: {start_date} - {end_date}")
    
    try:
        df = api.fetch_historical_data(
            symbol='BTC/USDT',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Ожидаемое количество свечей
        if timeframe == '15m':
            candles_per_day = 24 * 4  # 96 свечей в день (15 минут)
        elif timeframe == '1h':
            candles_per_day = 24
        elif timeframe == '1d':
            candles_per_day = 1
        else:
            candles_per_day = 96  # по умолчанию для 15m
        
        expected = candles_per_day * days
        actual = len(df)
        
        print(f"Ожидаемое количество свечей: {expected}")
        print(f"Фактическое количество свечей: {actual}")
        print(f"Разница: {actual - expected}")
        
        if actual >= expected * 0.9 and actual <= expected * 1.1:
            print("OK: количество свечей в допустимом диапазоне.")
        else:
            print("WARNING: количество свечей значительно отличается от ожидаемого.")
        
        # Вывод первых и последних строк
        if not df.empty:
            print(f"Первая свеча: {df.index[0]}")
            print(f"Последняя свеча: {df.index[-1]}")
        else:
            print("ERROR: Данные не загружены.")
            
    except Exception as e:
        print(f"ERROR при загрузке данных: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Тестируем разные периоды
    test_fetch_historical_data(7, '15m')
    test_fetch_historical_data(30, '15m')
    # Для 365 дней можно пропустить, так как загрузка займёт много времени
    # test_fetch_historical_data(365, '15m')
    print("\n=== Тест завершён ===")