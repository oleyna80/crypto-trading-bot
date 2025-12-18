"""
Тесты для модуля TrendDetector v2.0.
"""

import pytest
import pandas as pd
import numpy as np
from src.market_analysis.trend_detector import TrendDetector


class TestTrendDetector:
    """Тесты для класса TrendDetector."""
    
    @pytest.fixture
    def sample_daily_data(self):
        """Фикстура с тестовыми дневными данными."""
        np.random.seed(42)
        n = 100  # Достаточно для BB с периодом 24
        
        # Создаем базовый тренд
        trend = np.linspace(100, 110, n)
        
        # Добавляем волатильность
        volatility = np.random.normal(0, 2, n)
        
        close = trend + volatility
        high = close + np.random.uniform(0, 1, n)
        low = close - np.random.uniform(0, 1, n)
        open_price = close - np.random.uniform(-0.5, 0.5, n)
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='D'))
        
        return df
    
    @pytest.fixture
    def uptrend_data(self):
        """Фикстура с явным восходящим трендом."""
        n = 50
        # Сильный восходящий тренд
        close = np.linspace(100, 150, n)
        # Делаем цены выше верхней полосы BB
        close[-10:] = close[-10:] + 10  # Явно выше
        
        high = close + 2
        low = close - 2
        open_price = close - 1
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='D'))
        
        return df
    
    @pytest.fixture
    def downtrend_data(self):
        """Фикстура с явным нисходящим трендом."""
        n = 50
        # Сильный нисходящий тренд
        close = np.linspace(150, 100, n)
        # Делаем цены ниже нижней полосы BB
        close[-10:] = close[-10:] - 10  # Явно ниже
        
        high = close + 2
        low = close - 2
        open_price = close + 1
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='D'))
        
        return df
    
    @pytest.fixture
    def range_data(self):
        """Фикстура с боковым движением (range)."""
        n = 50
        # Боковое движение
        close = 100 + np.random.normal(0, 1, n)
        
        high = close + 1
        low = close - 1
        open_price = close - 0.5
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='D'))
        
        return df
    
    def test_initialization_valid(self, sample_daily_data):
        """Тест корректной инициализации."""
        td = TrendDetector(sample_daily_data, bb_period=24, fractal_lookback=2)
        assert td.daily_data is not None
        assert td.bb_period == 24
        assert td.fractal_lookback == 2
    
    def test_initialization_missing_columns(self):
        """Тест инициализации с отсутствующими колонками."""
        df = pd.DataFrame({'close': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Данные должны содержать колонки"):
            TrendDetector(df)
    
    def test_detect_trend_uptrend(self, uptrend_data):
        """Тест детекции восходящего тренда (UPTREND)."""
        td = TrendDetector(uptrend_data, bb_period=20)
        trend = td.detect_trend()
        
        # Должен определить UPTREND (цена выше верхней полосы)
        assert trend == "UPTREND"
    
    def test_detect_trend_downtrend(self, downtrend_data):
        """Тест детекции нисходящего тренда (DOWNTREND)."""
        td = TrendDetector(downtrend_data, bb_period=20)
        trend = td.detect_trend()
        
        # Должен определить DOWNTREND (цена ниже нижней полосы)
        assert trend == "DOWNTREND"
    
    def test_detect_trend_range(self, range_data):
        """Тест детекции бокового движения (RANGE)."""
        td = TrendDetector(range_data, bb_period=20)
        trend = td.detect_trend()
        
        # Должен определить RANGE (цена внутри полос)
        assert trend == "RANGE"
    
    def test_detect_trend_insufficient_data(self):
        """Тест детекции тренда при недостаточном количестве данных."""
        # Создаем данные с меньшим количеством точек, чем период BB
        df = pd.DataFrame({
            'open': [1, 2, 3, 4, 5],
            'high': [1.1, 2.1, 3.1, 4.1, 5.1],
            'low': [0.9, 1.9, 2.9, 3.9, 4.9],
            'close': [1, 2, 3, 4, 5]
        })
        
        td = TrendDetector(df, bb_period=10)  # Период больше количества данных
        trend = td.detect_trend()
        
        # Должен вернуть RANGE как fallback
        assert trend == "RANGE"
    
    def test_detect_trend_with_fractal_confirmation(self):
        """Тест детекции тренда с подтверждением фракталами."""
        # Создаем данные с явным фракталом и ценой выше верхней полосы
        n = 30
        close = np.ones(n) * 100
        
        # Делаем последнюю цену выше верхней полосы
        close[-1] = 120
        
        # Создаем явный фрактал вверх на индексе 20
        high = close.copy()
        high[20] = 115  # Фрактал вверх
        
        low = close - 2
        open_price = close - 1
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
        
        td = TrendDetector(df, bb_period=10)
        trend = td.detect_trend()
        
        # Должен определить UPTREND
        assert trend == "UPTREND"
    
    def test_is_fractal_broken_up(self, sample_daily_data):
        """Тест проверки пробития Up Fractal."""
        td = TrendDetector(sample_daily_data)
        
        # Создаем тестовый фрактал
        up_fractal = {
            'price': 100.0,
            'index': 10,
            'timestamp': pd.Timestamp('2024-01-01'),
            'confirmed': True
        }
        
        # Текущий low выше цены фрактала - фрактал пробит
        current_low = 101.0
        assert td._is_fractal_broken_up(up_fractal, current_low) == True
        
        # Текущий low ниже цены фрактала - фрактал не пробит
        current_low = 99.0
        assert td._is_fractal_broken_up(up_fractal, current_low) == False
    
    def test_is_fractal_broken_down(self, sample_daily_data):
        """Тест проверки пробития Down Fractal."""
        td = TrendDetector(sample_daily_data)
        
        # Создаем тестовый фрактал
        down_fractal = {
            'price': 100.0,
            'index': 10,
            'timestamp': pd.Timestamp('2024-01-01'),
            'confirmed': True
        }
        
        # Текущий high ниже цены фрактала - фрактал пробит
        current_high = 99.0
        assert td._is_fractal_broken_down(down_fractal, current_high) == True
        
        # Текущий high выше цены фрактала - фрактал не пробит
        current_high = 101.0
        assert td._is_fractal_broken_down(down_fractal, current_high) == False
    
    def test_get_detailed_analysis(self, sample_daily_data):
        """Тест получения детального анализа."""
        td = TrendDetector(sample_daily_data, bb_period=24)
        analysis = td.get_detailed_analysis()
        
        # Проверяем структуру результата
        assert isinstance(analysis, dict)
        required_keys = {
            'trend', 'close_price', 'bb_upper', 'bb_lower', 'bb_middle',
            'price_relative_to_bb', 'last_up_fractal', 'last_down_fractal',
            'up_fractals_count', 'down_fractals_count', 'timestamp'
        }
        assert set(analysis.keys()) == required_keys
        
        # Проверяем типы значений
        assert analysis['trend'] in ["UPTREND", "DOWNTREND", "RANGE"]
        assert isinstance(analysis['close_price'], (float, np.floating))
        assert isinstance(analysis['bb_upper'], (float, np.floating))
        assert isinstance(analysis['bb_lower'], (float, np.floating))
        assert isinstance(analysis['bb_middle'], (float, np.floating))
        assert isinstance(analysis['price_relative_to_bb'], (float, np.floating))
        assert isinstance(analysis['up_fractals_count'], int)
        assert isinstance(analysis['down_fractals_count'], int)
        
        # Проверяем логику: верхняя полоса >= средняя >= нижняя полоса
        assert analysis['bb_upper'] >= analysis['bb_middle']
        assert analysis['bb_middle'] >= analysis['bb_lower']
    
    def test_fallback_logic_bb_only(self):
        """Тест fallback логики при ошибке расчета фракталов."""
        # Создаем данные, где фракталы не будут найдены (мало данных)
        df = pd.DataFrame({
            'open': [1, 2, 3, 4, 5],
            'high': [1.1, 2.1, 3.1, 4.1, 5.1],
            'low': [0.9, 1.9, 2.9, 3.9, 4.9],
            'close': [1, 2, 3, 4, 5]
        })
        
        td = TrendDetector(df, bb_period=3)
        
        # Мокаем ошибку в find_fractals (но в реальности она произойдет из-за недостатка данных)
        # В данном тесте просто проверяем, что функция не падает
        trend = td.detect_trend()
        
        # Должен вернуть один из допустимых трендов
        assert trend in ["UPTREND", "DOWNTREND", "RANGE"]
    
    def test_edge_cases_price_exactly_on_band(self):
        """Тест граничного случая, когда цена точно на полосе."""
        n = 30
        # Создаем данные, где цена закрытия равна верхней полосе
        close = np.ones(n) * 100
        
        # Рассчитываем BB вручную для периода 10
        # SMA = 100, std = 0, верхняя полоса = 100
        # Значит цена равна верхней полосе
        
        high = close + 1
        low = close - 1
        open_price = close - 0.5
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
        
        td = TrendDetector(df, bb_period=10)
        trend = td.detect_trend()
        
        # Цена не выше верхней полосы (она равна), значит должна быть внутри
        # Согласно логике: price_above_upper = latest_close > upper_1sigma
        # Если равно, то False, значит price_inside = True
        assert trend == "RANGE"
    
    def test_different_bb_periods(self, sample_daily_data):
        """Тест работы с разными периодами BB."""
        for period in [10, 20, 30]:
            td = TrendDetector(sample_daily_data, bb_period=period)
            trend = td.detect_trend()
            
            # Должен вернуть один из допустимых трендов
            assert trend in ["UPTREND", "DOWNTREND", "RANGE"]
            
            # Детальный анализ должен работать
            analysis = td.get_detailed_analysis()
            assert analysis['trend'] == trend
    
    def test_trend_consistency(self):
        """Тест согласованности результатов при многократном вызове."""
        np.random.seed(123)
        n = 50
        close = 100 + np.random.normal(0, 5, n)
        
        df = pd.DataFrame({
            'open': close - 0.5,
            'high': close + 1,
            'low': close - 1,
            'close': close
        })
        
        td = TrendDetector(df, bb_period=20)
        
        # Многократный вызов должен давать одинаковый результат
        trend1 = td.detect_trend()
        trend2 = td.detect_trend()
        trend3 = td.detect_trend()
        
        assert trend1 == trend2 == trend3
        
        # Детальный анализ также должен быть согласован
        analysis1 = td.get_detailed_analysis()
        analysis2 = td.get_detailed_analysis()
        
        assert analysis1['trend'] == analysis2['trend']
        assert analysis1['close_price'] == analysis2['close_price']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])