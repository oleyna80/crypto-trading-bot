"""
Интеграционные тесты для Grid Trading Bot v2.0.

Тестирует полную цепочку автонастройки и работу основных классов v2.0:
- AdaptiveGridStrategy.auto_configure()
- AdaptiveBacktester.run_with_auto_config()
- Взаимодействие всех модулей автонастройки
"""

import pytest
import pandas as pd
import numpy as np
from src.grid_strategy_v2 import AdaptiveGridStrategy
from src.backtester_v2 import AdaptiveBacktester


class TestIntegrationV2:
    """Интеграционные тесты для v2.0 модулей."""
    
    @pytest.fixture
    def sample_daily_data(self):
        """Фикстура с тестовыми дневными данными."""
        np.random.seed(42)
        n = 100
        
        # Создаем реалистичные данные с трендом и волатильностью
        trend = np.linspace(50000, 55000, n)
        noise = np.random.normal(0, 1000, n)
        
        close = trend + noise
        high = close + np.random.uniform(0, 500, n)
        low = close - np.random.uniform(0, 500, n)
        open_price = close - np.random.uniform(-200, 200, n)
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='D'))
        
        return df
    
    @pytest.fixture
    def sample_intraday_data(self):
        """Фикстура с тестовыми внутридневными данными (15m)."""
        np.random.seed(42)
        n = 500  # Больше данных для внутридневной торговли
        
        # Создаем более волатильные данные
        base = np.linspace(52000, 53000, n)
        noise = np.random.normal(0, 200, n)
        
        close = base + noise
        high = close + np.random.uniform(0, 100, n)
        low = close - np.random.uniform(0, 100, n)
        open_price = close - np.random.uniform(-50, 50, n)
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(500, 5000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
        
        return df
    
    @pytest.fixture
    def sample_balance(self):
        """Фикстура с тестовым балансом."""
        return 10000.0  # 10,000 USDT
    
    def test_adaptive_grid_strategy_auto_configure_full_chain(self, sample_daily_data, sample_intraday_data, sample_balance):
        """Тест полной цепочки автонастройки AdaptiveGridStrategy."""
        # 1. Создание стратегии
        strategy = AdaptiveGridStrategy(deposit=sample_balance)
        assert strategy is not None
        assert not strategy.is_auto_configured()
        
        # 2. Автонастройка
        config_params = strategy.auto_configure(
            daily_data=sample_daily_data,
            data_15m=sample_intraday_data,
            balance=sample_balance
        )
        
        # 3. Проверка результатов автонастройки
        assert strategy.is_auto_configured()
        assert config_params is not None
        assert isinstance(config_params, dict)
        
        # Проверка обязательных полей
        required_fields = [
            'mode', 'upper_bound', 'lower_bound', 'num_levels',
            'amount_per_level', 'grid_step', 'volatility', 'current_price'
        ]
        for field in required_fields:
            assert field in config_params, f"Отсутствует поле {field} в конфигурации"
        
        # Проверка логических ограничений
        assert config_params['upper_bound'] > config_params['lower_bound']
        assert config_params['num_levels'] >= 10  # Минимум по GridLevelsCalculator
        assert config_params['num_levels'] <= 60  # Максимум по GridLevelsCalculator
        assert config_params['amount_per_level'] > 0
        assert config_params['volatility'] > 0
        assert config_params['current_price'] > 0
        
        # 4. Проверка, что стратегия правильно обновлена
        assert strategy.upper_bound == config_params['upper_bound']
        assert strategy.lower_bound == config_params['lower_bound']
        assert strategy.num_levels == config_params['num_levels']
        assert strategy.amount_per_level == config_params['amount_per_level']
        assert strategy.mode == config_params['mode']
        
        # 5. Проверка метода get_configuration
        retrieved_config = strategy.get_configuration()
        assert retrieved_config == config_params
        
        # 6. Проверка расчета уровней
        levels = strategy.calculate_levels(current_price=config_params['current_price'])
        assert isinstance(levels, list)
        assert len(levels) == config_params['num_levels']
        
        # Уровни должны быть отсортированы (сравниваем цены уровней)
        # GridLevel объекты имеют атрибут price
        level_prices = [level.price for level in levels]
        assert all(level_prices[i] <= level_prices[i+1] for i in range(len(level_prices)-1))
        
        # Первый уровень должен быть близок к lower_bound, последний - к upper_bound
        assert abs(level_prices[0] - config_params['lower_bound']) < 1.0
        assert abs(level_prices[-1] - config_params['upper_bound']) < 1.0
    
    def test_adaptive_grid_strategy_fallback_configuration(self, sample_daily_data, sample_balance):
        """Тест fallback конфигурации при ошибке."""
        # Создаем некорректные данные для вызова ошибки
        invalid_data = pd.DataFrame({
            'open': [1, 2, 3],  # Слишком мало данных
            'high': [1.1, 2.1, 3.1],
            'low': [0.9, 1.9, 2.9],
            'close': [1, 2, 3]
        })
        
        strategy = AdaptiveGridStrategy(deposit=sample_balance)
        
        # Используем невалидные данные для вызова fallback
        config_params = strategy.auto_configure(
            daily_data=invalid_data,
            data_15m=invalid_data,  # Тоже невалидные
            balance=sample_balance
        )
        
        # Должен вернуть fallback конфигурацию
        assert config_params is not None
        assert 'fallback' in config_params
        assert config_params.get('fallback') == True
        
        # Проверка, что стратегия имеет базовые параметры
        assert strategy.upper_bound > strategy.lower_bound
        assert strategy.num_levels > 0
        assert strategy.amount_per_level > 0
    
    def test_adaptive_backtester_run_with_auto_config(self, sample_daily_data, sample_intraday_data, sample_balance):
        """Тест запуска бэктеста с автонастройкой."""
        # 1. Создание бэктестера
        backtester = AdaptiveBacktester(
            initial_balance=sample_balance,
            use_auto_config=True
        )
        assert backtester is not None
        assert backtester.use_auto_config == True
        
        # 2. Запуск бэктеста с автонастройкой
        results = backtester.run_with_auto_config(
            daily_data=sample_daily_data,
            intraday_data=sample_intraday_data,
            fee_rate=0.0006
        )
        
        # 3. Проверка результатов
        assert results is not None
        assert isinstance(results, dict)
        
        # Проверка обязательных полей в результатах (могут быть вложены)
        # run_with_auto_config возвращает расширенные метрики
        assert 'configuration' in results
        assert 'trading_mode' in results
        assert 'use_auto_config' in results
        
        # Проверяем конфигурацию
        config = results['configuration']
        assert config is not None
        assert 'mode' in config
        assert 'num_levels' in config
        assert 'upper_bound' in config
        assert 'lower_bound' in config
        
        # Проверяем, что стратегия создана и настроена
        strategy = backtester.get_adaptive_strategy()
        assert strategy is not None
        assert isinstance(strategy, AdaptiveGridStrategy)
        
        config = results['configuration']
        assert config is not None
        assert 'mode' in config
        assert 'num_levels' in config
        assert 'upper_bound' in config
        assert 'lower_bound' in config
        
        # 4. Проверка, что стратегия создана и настроена
        strategy = backtester.get_adaptive_strategy()
        assert strategy is not None
        assert isinstance(strategy, AdaptiveGridStrategy)
        assert strategy.is_auto_configured()
        
        # 5. Проверка истории конфигураций
        config_history = backtester.get_configuration_history()
        assert isinstance(config_history, list)
        assert len(config_history) > 0
        
        # Последняя конфигурация должна совпадать с текущей
        last_config = config_history[-1]['params']
        assert last_config['mode'] == config['mode']
        assert last_config['num_levels'] == config['num_levels']
    
    def test_adaptive_backtester_standard_vs_auto_config(self, sample_intraday_data, sample_balance):
        """Тест сравнения стандартного бэктеста и бэктеста с автонастройкой."""
        # 1. Бэктест с автонастройкой
        auto_backtester = AdaptiveBacktester(
            initial_balance=sample_balance,
            use_auto_config=True
        )
        
        # Для автонастройки нужны дневные данные, ресемплируем
        daily_data = sample_intraday_data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        auto_results = auto_backtester.run_with_auto_config(
            daily_data=daily_data,
            intraday_data=sample_intraday_data,
            fee_rate=0.0006
        )
        
        # 2. Бэктест со стандартной стратегией
        standard_backtester = AdaptiveBacktester(
            initial_balance=sample_balance,
            use_auto_config=False
        )
        
        # Создаем простую стратегию для сравнения
        from models.grid_strategy import GridStrategy
        current_price = sample_intraday_data['close'].iloc[-1]
        standard_strategy = GridStrategy(
            upper_bound=current_price * 1.05,
            lower_bound=current_price * 0.95,
            num_levels=20,
            amount_per_level=0.001,
            deposit=sample_balance
        )
        
        standard_results = standard_backtester.run_backtest(
            data=sample_intraday_data,
            strategy=standard_strategy,
            fee_rate=0.0006
        )
        
        # 3. Проверка, что оба бэктеста завершились успешно
        assert 'final_equity' in auto_results
        assert 'final_equity' in standard_results
        
        # 4. Проверка, что автонастройка добавила дополнительные метрики
        assert 'configuration' in auto_results
        assert 'trading_mode' in auto_results
        assert 'roi_on_investment' in auto_results
        assert 'capital_utilization_percent' in auto_results
        
        # В стандартных результатах этих полей не должно быть
        assert 'configuration' not in standard_results
        assert 'trading_mode' not in standard_results
    
    def test_adaptive_backtester_fallback_mode(self, sample_balance):
        """Тест fallback режима бэктестера при ошибках."""
        # Создаем некорректные данные
        invalid_data = pd.DataFrame({
            'open': [1],
            'high': [1.1],
            'low': [0.9],
            'close': [1]
        })
        
        backtester = AdaptiveBacktester(
            initial_balance=sample_balance,
            use_auto_config=True
        )
        
        # Запуск с некорректными данными должен вызвать fallback
        results = backtester.run_with_auto_config(
            daily_data=invalid_data,
            intraday_data=invalid_data,
            fee_rate=0.0006
        )
        
        # Должен вернуть результаты (может быть fallback или обычная конфигурация)
        assert results is not None
        
        # Должна быть информация о конфигурации
        assert 'configuration' in results
        
        # Проверяем, что конфигурация содержит необходимые поля
        config = results['configuration']
        assert 'mode' in config
        assert 'upper_bound' in config
        assert 'lower_bound' in config
        
        # Fallback может быть отмечен или нет, в зависимости от реализации
        # Главное - что функция не упала и вернула результаты
    
    def test_configuration_comparison(self, sample_balance):
        """Тест сравнения конфигураций."""
        backtester = AdaptiveBacktester(
            initial_balance=sample_balance,
            use_auto_config=False
        )
        
        # Создаем две тестовые конфигурации
        config1 = {
            'mode': 'RANGE',
            'upper_bound': 55000.0,
            'lower_bound': 45000.0,
            'num_levels': 25,
            'volatility': 0.02
        }
        
        config2 = {
            'mode': 'UPTREND',
            'upper_bound': 60000.0,
            'lower_bound': 50000.0,
            'num_levels': 20,
            'volatility': 0.015
        }
        
        # Сравниваем конфигурации
        comparison = backtester.compare_configurations(config1, config2)
        
        assert comparison is not None
        assert isinstance(comparison, dict)
        
        # Проверяем сравнение для числовых полей
        assert 'upper_bound' in comparison
        assert 'difference' in comparison['upper_bound']
        assert 'difference_percent' in comparison['upper_bound']
        
        # Проверяем сравнение для строковых полей
        assert 'mode' in comparison
        assert 'different' in comparison['mode']
        assert comparison['mode']['different'] == True
        
        # Проверяем вычисления
        upper_diff = comparison['upper_bound']['difference']
        assert upper_diff == 5000.0  # 60000 - 55000
    
    def test_adaptive_strategy_edge_cases(self, sample_daily_data, sample_intraday_data, sample_balance):
        """Тест граничных случаев AdaptiveGridStrategy."""
        # 1. Стратегия без депозита
        strategy_no_deposit = AdaptiveGridStrategy()
        assert strategy_no_deposit.deposit == 1000.0  # Значение по умолчанию
        
        # 2. Стратегия с частичными параметрами
        strategy_partial = AdaptiveGridStrategy(
            upper_bound=55000.0,
            lower_bound=45000.0,
            deposit=sample_balance
        )
        # Остальные параметры должны быть установлены по умолчанию
        assert strategy_partial.num_levels == 1
        assert strategy_partial.amount_per_level == 0.001
        
        # 3. Стратегия с полными параметрами (без автонастройки)
        strategy_full = AdaptiveGridStrategy(
            upper_bound=55000.0,
            lower_bound=45000.0,
            num_levels=20,
            amount_per_level=0.001,
            deposit=sample_balance
        )
        assert not strategy_full.is_auto_configured()
        assert strategy_full.mode == "RANGE"  # Значение по умолчанию
        
        # 4. Проверка строкового представления
        str_repr = str(strategy_full)
        assert "AdaptiveGridStrategy" in str_repr
        assert "mode=RANGE" in str_repr
    
    def test_volatility_module_integration(self, sample_daily_data):
        """Тест интеграции с модулем волатильности."""
        from src.market_analysis.volatility import VolatilityCalculator
        
        # Расчет волатильности
        volatility = VolatilityCalculator.calculate_historical_volatility(
            sample_daily_data['close'], period=30
        )
        
        assert volatility > 0
        assert isinstance(volatility, float)
        
        # Проверка, что волатильность используется в автонастройке
        strategy = AdaptiveGridStrategy(deposit=10000.0)
        config = strategy.auto_configure(
            daily_data=sample_daily_data,
            data_15m=sample_daily_data,  # Используем те же данные для простоты
            balance=10000.0
        )
        
        assert 'volatility' in config
        assert config['volatility'] > 0
    
    def test_trend_detector_integration(self, sample_daily_data):
        """Тест интеграции с детектором тренда."""
        from src.market_analysis.trend_detector import TrendDetector
        
        detector = TrendDetector(sample_daily_data)
        trend = detector.detect_trend()
        
        # Должен вернуть один из допустимых трендов
        assert trend in ["UPTREND", "DOWNTREND", "RANGE"]
        
        # Проверка детального анализа
        analysis = detector.get_detailed_analysis()
        assert analysis['trend'] == trend
        assert 'close_price' in analysis
        assert 'bb_upper' in analysis
        assert 'bb_lower' in analysis


if __name__ == '__main__':
    pytest.main([__file__, '-v'])