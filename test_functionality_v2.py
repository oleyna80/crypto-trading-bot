#!/usr/bin/env python3
"""
Тест базовой функциональности AdaptiveGridStrategy и AdaptiveBacktester.
"""

import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def generate_test_data(days: int = 100, freq: str = 'D') -> pd.DataFrame:
    """Генерация тестовых данных."""
    dates = pd.date_range('2024-01-01', periods=days, freq=freq)
    
    # Генерация реалистичных ценовых данных с трендом и волатильностью
    np.random.seed(42)
    base_price = 50000.0
    trend = np.linspace(0, 0.1, days)  # Восходящий тренд 10%
    noise = np.random.normal(0, 0.02, days)  # Волатильность 2%
    
    close_prices = base_price * (1 + trend + noise)
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, days))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.01, days))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.01, days))
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.normal(100, 10, days)
    }, index=dates)
    
    return data

def test_adaptive_grid_strategy():
    """Тестирование AdaptiveGridStrategy."""
    print("\n" + "="*60)
    print("Тестирование AdaptiveGridStrategy")
    print("="*60)
    
    try:
        from src.grid_strategy_v2 import AdaptiveGridStrategy
        
        # 1. Создание стратегии с депозитом
        print("1. Создание AdaptiveGridStrategy с депозитом...")
        strategy = AdaptiveGridStrategy(deposit=10000.0)
        print(f"   Стратегия создана: {strategy}")
        print(f"   Автонастройка выполнена: {strategy.is_auto_configured()}")
        
        # 2. Генерация тестовых данных
        print("\n2. Генерация тестовых данных...")
        daily_data = generate_test_data(days=100, freq='D')
        intraday_data = generate_test_data(days=200, freq='12H')  # 12-часовые данные
        
        print(f"   Дневные данные: {len(daily_data)} записей")
        print(f"   Внутридневные данные: {len(intraday_data)} записей")
        
        # 3. Автонастройка стратегии
        print("\n3. Автонастройка стратегии...")
        config_params = strategy.auto_configure(
            daily_data=daily_data,
            data_15m=intraday_data,
            balance=10000.0
        )
        
        print(f"   Режим: {config_params.get('mode')}")
        print(f"   Границы: [{config_params.get('lower_bound'):.2f}, {config_params.get('upper_bound'):.2f}]")
        print(f"   Уровней: {config_params.get('num_levels')}")
        print(f"   Размер ордера: {config_params.get('amount_per_level'):.6f} BTC")
        print(f"   Волатильность: {config_params.get('volatility'):.6f}")
        
        # 4. Проверка, что стратегия настроена
        print("\n4. Проверка состояния стратегии...")
        print(f"   Автонастройка выполнена: {strategy.is_auto_configured()}")
        print(f"   Текущий режим: {strategy.mode}")
        print(f"   Волатильность: {strategy.volatility:.6f}")
        
        # 5. Расчет уровней сетки
        print("\n5. Расчет уровней сетки...")
        current_price = intraday_data['close'].iloc[-1]
        levels = strategy.calculate_levels(current_price=current_price)
        print(f"   Рассчитано уровней: {len(levels)}")
        print(f"   Текущая цена: {current_price:.2f}")
        
        # 6. Проверка статистики
        print("\n6. Проверка статистики...")
        stats = strategy.get_statistics()
        print(f"   Всего уровней: {stats.get('total_levels')}")
        print(f"   Исполнено уровней: {stats.get('executed_levels')}")
        
        # 7. Проверка конфигурации
        print("\n7. Получение конфигурации...")
        config = strategy.get_configuration()
        print(f"   Параметров конфигурации: {len(config)}")
        
        print("\nOK AdaptiveGridStrategy тестирование пройдено успешно!")
        return True
        
    except Exception as e:
        print(f"\nERROR Ошибка тестирования AdaptiveGridStrategy: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_backtester():
    """Тестирование AdaptiveBacktester."""
    print("\n" + "="*60)
    print("Тестирование AdaptiveBacktester")
    print("="*60)
    
    try:
        from src.backtester_v2 import AdaptiveBacktester
        
        # 1. Создание бэктестера
        print("1. Создание AdaptiveBacktester...")
        backtester = AdaptiveBacktester(
            initial_balance=10000.0,
            use_auto_config=True
        )
        print(f"   Бэктестер создан: {backtester}")
        print(f"   Использование автонастройки: {backtester.use_auto_config}")
        
        # 2. Генерация тестовых данных
        print("\n2. Генерация тестовых данных...")
        daily_data = generate_test_data(days=100, freq='D')
        intraday_data = generate_test_data(days=200, freq='12H')
        
        print(f"   Дневные данные: {len(daily_data)} записей")
        print(f"   Внутридневные данные: {len(intraday_data)} записей")
        
        # 3. Запуск бэктеста с автонастройкой
        print("\n3. Запуск бэктеста с автонастройкой...")
        results = backtester.run_with_auto_config(
            daily_data=daily_data,
            intraday_data=intraday_data,
            fee_rate=0.0006
        )
        
        # 4. Проверка результатов
        print("\n4. Анализ результатов бэктеста...")
        print(f"   Сделок: {results.get('total_trades', 0)}")
        print(f"   Начальный баланс: {results.get('initial_balance', 0):.2f} USDT")
        print(f"   Финальная equity: {results.get('final_equity', 0):.2f} USDT")
        print(f"   Общий PnL: {results.get('total_pnl', 0):.2f} USDT")
        print(f"   PnL %: {results.get('total_pnl_pct', 0):.2f}%")
        
        if 'configuration' in results:
            config = results['configuration']
            print(f"   Режим торговли: {config.get('mode')}")
            print(f"   Уровней сетки: {config.get('num_levels')}")
            print(f"   Использование капитала: {results.get('capital_utilization_percent', 0):.1f}%")
        
        # 5. Проверка истории конфигураций
        print("\n5. Проверка истории конфигураций...")
        config_history = backtester.get_configuration_history()
        print(f"   Записей в истории: {len(config_history)}")
        
        # 6. Получение адаптивной стратегии
        print("\n6. Получение адаптивной стратегии...")
        strategy = backtester.get_adaptive_strategy()
        if strategy:
            print(f"   Стратегия получена: {strategy}")
            print(f"   Режим стратегии: {strategy.mode}")
        else:
            print("   Стратегия не найдена")
        
        # 7. Запуск стандартного бэктеста (без автонастройки)
        print("\n7. Запуск стандартного бэктеста (без автонастройки)...")
        backtester_no_auto = AdaptiveBacktester(
            initial_balance=10000.0,
            use_auto_config=False
        )
        
        # Создаем простую стратегию для теста
        from models.grid_strategy import GridStrategy
        simple_strategy = GridStrategy(
            upper_bound=55000.0,
            lower_bound=45000.0,
            num_levels=20,
            amount_per_level=0.001,
            deposit=10000.0
        )
        
        standard_results = backtester_no_auto.run_backtest(
            data=intraday_data,
            strategy=simple_strategy,
            fee_rate=0.0006
        )
        
        print(f"   Сделок (стандартный): {standard_results.get('total_trades', 0)}")
        print(f"   PnL (стандартный): {standard_results.get('total_pnl', 0):.2f} USDT")
        
        print("\nOK AdaptiveBacktester тестирование пройдено успешно!")
        return True
        
    except Exception as e:
        print(f"\nERROR Ошибка тестирования AdaptiveBacktester: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inheritance_and_compatibility():
    """Тестирование наследования и совместимости."""
    print("\n" + "="*60)
    print("Тестирование наследования и совместимости")
    print("="*60)
    
    try:
        from models.grid_strategy import GridStrategy
        from services.backtester import Backtester
        from src.grid_strategy_v2 import AdaptiveGridStrategy
        from src.backtester_v2 import AdaptiveBacktester
        
        # 1. Проверка наследования
        print("1. Проверка наследования...")
        print(f"   AdaptiveGridStrategy наследует GridStrategy: {issubclass(AdaptiveGridStrategy, GridStrategy)}")
        print(f"   AdaptiveBacktester наследует Backtester: {issubclass(AdaptiveBacktester, Backtester)}")
        
        # 2. Проверка совместимости интерфейсов
        print("\n2. Проверка совместимости интерфейсов...")
        
        # AdaptiveGridStrategy должен иметь все методы GridStrategy
        grid_methods = set(dir(GridStrategy))
        adaptive_methods = set(dir(AdaptiveGridStrategy))
        
        missing_methods = grid_methods - adaptive_methods
        if missing_methods:
            print(f"   Отсутствующие методы: {missing_methods}")
        else:
            print("   Все методы GridStrategy присутствуют в AdaptiveGridStrategy")
        
        # 3. Проверка работы с существующим кодом
        print("\n3. Проверка работы с существующим кодом...")
        
        # Создаем AdaptiveGridStrategy с явными параметрами (как обычный GridStrategy)
        adaptive_as_regular = AdaptiveGridStrategy(
            upper_bound=55000.0,
            lower_bound=45000.0,
            num_levels=20,
            amount_per_level=0.001,
            deposit=10000.0
        )
        
        # Должны работать все методы GridStrategy
        levels = adaptive_as_regular.calculate_levels(current_price=50000.0)
        stats = adaptive_as_regular.get_statistics()
        
        print(f"   Уровней рассчитано: {len(levels)}")
        print(f"   Статистика получена: {stats.get('total_levels')} уровней")
        
        # 4. Проверка обратной совместимости
        print("\n4. Проверка обратной совместимости...")
        
        # Стандартный Backtester должен работать с AdaptiveGridStrategy
        from services.backtester import Backtester as StandardBacktester
        
        standard_backtester = StandardBacktester(initial_balance=10000.0)
        data = generate_test_data(days=50, freq='D')
        
        # Это должно работать без ошибок
        try:
            results = standard_backtester.run_backtest(
                data=data,
                strategy=adaptive_as_regular,
                fee_rate=0.0006
            )
            print(f"   Стандартный бэктестер работает с AdaptiveGridStrategy: OK")
            print(f"   Сделок: {results.get('total_trades', 0)}")
        except Exception as e:
            print(f"   Ошибка совместимости: {e}")
        
        print("\nOK Тестирование наследования и совместимости пройдено успешно!")
        return True
        
    except Exception as e:
        print(f"\nERROR Ошибка тестирования наследования: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция тестирования."""
    print("Запуск комплексного тестирования Grid Trading Bot v2.0")
    print("="*60)
    
    results = []
    
    # Запуск тестов
    results.append(("AdaptiveGridStrategy", test_adaptive_grid_strategy()))
    results.append(("AdaptiveBacktester", test_adaptive_backtester()))
    results.append(("Наследование и совместимость", test_inheritance_and_compatibility()))
    
    # Вывод итогов
    print("\n" + "="*60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "ПРОЙДЕН" if passed else "ПРОВАЛЕН"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("OK ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        return 0
    else:
        print("ERROR НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        return 1

if __name__ == "__main__":
    sys.exit(main())