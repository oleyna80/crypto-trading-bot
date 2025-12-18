"""
Тесты для модулей индикаторов v2.0:
- BollingerBands
- FractalDetector
"""

import pytest
import pandas as pd
import numpy as np
from src.indicators.bollinger_bands import BollingerBands
from src.indicators.fractals import FractalDetector


class TestBollingerBands:
    """Тесты для класса BollingerBands."""
    
    @pytest.fixture
    def sample_prices(self):
        """Фикстура с тестовыми ценами."""
        # Создаем детерминированный ряд цен с трендом
        np.random.seed(42)
        n = 100
        trend = np.linspace(100, 110, n)
        noise = np.random.normal(0, 2, n)
        prices = trend + noise
        return pd.Series(prices, index=pd.date_range('2024-01-01', periods=n, freq='h'))
    
    def test_initialization_valid(self, sample_prices):
        """Тест корректной инициализации."""
        bb = BollingerBands(sample_prices, period=20, num_std=2.0)
        assert bb.data is not None
        assert bb.period == 20
        assert bb.num_std == 2.0
    
    def test_initialization_invalid_data(self):
        """Тест инициализации с некорректными данными."""
        with pytest.raises(ValueError, match="Данные не могут быть пустыми"):
            BollingerBands(pd.Series([]))
        
        with pytest.raises(ValueError, match="Период должен быть больше 1"):
            BollingerBands(pd.Series([1, 2, 3]), period=1)
        
        with pytest.raises(ValueError, match="Количество стандартных отклонений должно быть положительным"):
            BollingerBands(pd.Series([1, 2, 3]), num_std=0)
    
    def test_calculate_sufficient_data(self, sample_prices):
        """Тест расчета с достаточным количеством данных."""
        bb = BollingerBands(sample_prices, period=20)
        result = bb.calculate()
        
        # Проверяем наличие всех ключей
        expected_keys = {'middle', 'upper_1sigma', 'lower_1sigma', 
                        'upper_2sigma', 'lower_2sigma', 'bandwidth'}
        assert set(result.keys()) == expected_keys
        
        # Проверяем, что все серии имеют правильную длину
        for key, series in result.items():
            assert len(series) == len(sample_prices)
            # Первые (period-1) значения должны быть NaN
            assert series.iloc[:19].isna().all()
            # Остальные значения не должны быть NaN
            assert not series.iloc[19:].isna().any()
        
        # Проверяем логику расчета
        middle = result['middle']
        upper_1sigma = result['upper_1sigma']
        lower_1sigma = result['lower_1sigma']
        
        # upper_1sigma должен быть >= middle
        assert (upper_1sigma.iloc[19:] >= middle.iloc[19:]).all()
        # lower_1sigma должен быть <= middle
        assert (lower_1sigma.iloc[19:] <= middle.iloc[19:]).all()
        
        # Проверяем ширину полос
        bandwidth = result['bandwidth']
        assert (bandwidth.iloc[19:] >= 0).all()
    
    def test_calculate_insufficient_data(self):
        """Тест расчета с недостаточным количеством данных."""
        prices = pd.Series([1, 2, 3, 4, 5])
        bb = BollingerBands(prices, period=10)
        
        with pytest.raises(ValueError, match="Недостаточно данных для периода"):
            bb.calculate()
    
    def test_get_last_values(self, sample_prices):
        """Тест получения последних значений."""
        bb = BollingerBands(sample_prices, period=20)
        last_values = bb.get_last_values()
        
        assert last_values is not None
        assert isinstance(last_values, dict)
        
        # Проверяем наличие всех ключей
        expected_keys = {'middle', 'upper_1sigma', 'lower_1sigma', 
                        'upper_2sigma', 'lower_2sigma', 'bandwidth'}
        assert set(last_values.keys()) == expected_keys
        
        # Проверяем, что значения являются числами
        for key, value in last_values.items():
            assert isinstance(value, (float, np.floating))
            # Проверяем, что значения не NaN (кроме возможных edge cases)
            assert not np.isnan(value)
        
        # Проверяем логику: upper_1sigma >= middle >= lower_1sigma
        assert last_values['upper_1sigma'] >= last_values['middle']
        assert last_values['middle'] >= last_values['lower_1sigma']
    
    def test_get_last_values_with_nan_data(self):
        """Тест получения последних значений с данными, содержащими NaN."""
        prices = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        bb = BollingerBands(prices, period=3)
        last_values = bb.get_last_values()
        
        # Метод может вернуть словарь с NaN значениями или None в зависимости от реализации
        # Проверяем, что возвращается либо None, либо словарь
        if last_values is not None:
            assert isinstance(last_values, dict)
            # Проверяем наличие ожидаемых ключей
            expected_keys = {'middle', 'upper_1sigma', 'lower_1sigma',
                            'upper_2sigma', 'lower_2sigma', 'bandwidth'}
            assert set(last_values.keys()) == expected_keys
        # Если вернулся None - это тоже допустимо
    
    def test_different_std_multipliers(self, sample_prices):
        """Тест с разными множителями стандартного отклонения."""
        for num_std in [1.0, 1.5, 2.0, 2.5]:
            bb = BollingerBands(sample_prices, period=20, num_std=num_std)
            result = bb.calculate()
            
            # Проверяем, что полосы на 2σ шире, чем на 1σ
            upper_1sigma = result['upper_1sigma'].iloc[50]
            upper_2sigma = result['upper_2sigma'].iloc[50]
            lower_1sigma = result['lower_1sigma'].iloc[50]
            lower_2sigma = result['lower_2sigma'].iloc[50]
            
            assert upper_2sigma >= upper_1sigma
            assert lower_2sigma <= lower_1sigma


class TestFractalDetector:
    """Тесты для класса FractalDetector."""
    
    @pytest.fixture
    def sample_ohlc(self):
        """Фикстура с тестовыми OHLC данными."""
        np.random.seed(42)
        n = 50
        base = np.linspace(100, 110, n)
        
        # Создаем искусственные фракталы
        highs = base + np.random.uniform(0, 2, n)
        lows = base - np.random.uniform(0, 2, n)
        
        # Добавляем явные фракталы вверх на индексах 10 и 30
        highs[10] = highs[10] + 5  # Явный максимум
        highs[30] = highs[30] + 5
        
        # Добавляем явные фракталы вниз на индексах 20 и 40
        lows[20] = lows[20] - 5    # Явный минимум
        lows[40] = lows[40] - 5
        
        df = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': (highs + lows) / 2,
            'open': (highs + lows) / 2 - 0.5,
            'volume': np.random.randint(100, 1000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        
        return df
    
    def test_initialization_valid(self, sample_ohlc):
        """Тест корректной инициализации."""
        fd = FractalDetector(sample_ohlc, lookback=2)
        assert fd.data is not None
        assert fd.lookback == 2
    
    def test_initialization_missing_columns(self):
        """Тест инициализации с отсутствующими колонками."""
        df = pd.DataFrame({'close': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Данные должны содержать колонки"):
            FractalDetector(df)
    
    def test_find_fractals_sufficient_data(self, sample_ohlc):
        """Тест обнаружения фракталов с достаточным количеством данных."""
        fd = FractalDetector(sample_ohlc, lookback=2)
        fractals = fd.find_fractals()
        
        # Проверяем структуру результата
        assert isinstance(fractals, dict)
        assert 'up_fractals' in fractals
        assert 'down_fractals' in fractals
        assert 'last_up' in fractals
        assert 'last_down' in fractals
        
        # Проверяем, что списки фракталов не пусты (должны найти наши искусственные)
        assert len(fractals['up_fractals']) >= 2
        assert len(fractals['down_fractals']) >= 2
        
        # Проверяем структуру информации о фракталах
        for fractal in fractals['up_fractals']:
            assert 'index' in fractal
            assert 'timestamp' in fractal
            assert 'price' in fractal
            assert 'confirmed' in fractal
            assert isinstance(fractal['index'], int)
            assert isinstance(fractal['price'], float)
            assert isinstance(fractal['confirmed'], bool)
        
        # Проверяем, что последние фракталы корректны
        if fractals['last_up']:
            assert fractals['last_up'] == fractals['up_fractals'][-1]
        if fractals['last_down']:
            assert fractals['last_down'] == fractals['down_fractals'][-1]
    
    def test_find_fractals_insufficient_data(self):
        """Тест обнаружения фракталов с недостаточным количеством данных."""
        df = pd.DataFrame({
            'high': [1, 2, 3, 4],
            'low': [0.5, 1.5, 2.5, 3.5]
        })
        
        fd = FractalDetector(df, lookback=2)
        fractals = fd.find_fractals()
        
        # Должен вернуть пустые списки
        assert fractals['up_fractals'] == []
        assert fractals['down_fractals'] == []
        assert fractals['last_up'] is None
        assert fractals['last_down'] is None
    
    def test_is_up_fractal_logic(self):
        """Тест логики определения фрактала вверх."""
        # Создаем данные с явным фракталом вверх
        highs = np.array([10, 11, 12, 15, 13, 14, 11])  # Индекс 3 - фрактал (15 > всех соседей)
        
        fd = FractalDetector(pd.DataFrame({'high': highs, 'low': highs - 1}))
        
        # Индекс 3 должен быть фракталом вверх
        assert fd._is_up_fractal(3, highs) == True
        
        # Индекс 2 не должен быть фракталом (12 < 15 справа)
        assert fd._is_up_fractal(2, highs) == False
    
    def test_is_down_fractal_logic(self):
        """Тест логики определения фрактала вниз."""
        # Создаем данные с явным фракталом вниз
        lows = np.array([15, 14, 13, 10, 12, 13, 14])  # Индекс 3 - фрактал (10 < всех соседей)
        
        fd = FractalDetector(pd.DataFrame({'high': lows + 1, 'low': lows}))
        
        # Индекс 3 должен быть фракталом вниз
        assert fd._is_down_fractal(3, lows) == True
        
        # Индекс 2 не должен быть фракталом (13 > 10 справа)
        assert fd._is_down_fractal(2, lows) == False
    
    def test_confirm_fractal(self, sample_ohlc):
        """Тест подтверждения фракталов."""
        fd = FractalDetector(sample_ohlc, lookback=2)
        
        # Тестируем подтверждение фрактала вверх
        # Создаем искусственную ситуацию
        test_idx = 10
        confirmed = fd._confirm_fractal(test_idx, direction='up')
        
        # Результат зависит от данных, но функция должна вернуть bool
        assert isinstance(confirmed, bool)
        
        # Тестируем подтверждение фрактала вниз
        confirmed = fd._confirm_fractal(test_idx, direction='down')
        assert isinstance(confirmed, bool)
        
        # Тестируем некорректное направление
        with pytest.raises(ValueError, match="Неизвестное направление"):
            fd._confirm_fractal(test_idx, direction='invalid')
    
    def test_get_fractal_zones(self, sample_ohlc):
        """Тест получения зон фракталов."""
        fd = FractalDetector(sample_ohlc, lookback=2)
        zones = fd.get_fractal_zones()
        
        assert isinstance(zones, dict)
        assert 'up_zones' in zones
        assert 'down_zones' in zones
        
        # Проверяем, что зоны содержат кортежи (индекс, цена)
        for zone_list in zones.values():
            assert isinstance(zone_list, list)
            if zone_list:
                for zone in zone_list:
                    assert isinstance(zone, tuple)
                    assert len(zone) == 2
                    assert isinstance(zone[0], int)
                    assert isinstance(zone[1], float)
    
    def test_edge_cases(self):
        """Тест граничных случаев."""
        # Данные с одинаковыми значениями (не должно быть фракталов)
        n = 10
        df = pd.DataFrame({
            'high': np.full(n, 100.0),
            'low': np.full(n, 99.0)
        })
        
        fd = FractalDetector(df, lookback=2)
        fractals = fd.find_fractals()
        
        # Не должно быть фракталов при одинаковых значениях
        assert len(fractals['up_fractals']) == 0
        assert len(fractals['down_fractals']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])