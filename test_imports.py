#!/usr/bin/env python3
"""
Тест импортов для AdaptiveGridStrategy и AdaptiveBacktester.
"""

import sys
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_imports():
    """Тестирование импортов всех необходимых модулей."""
    print("Тестирование импортов для Grid Trading Bot v2.0...")
    
    try:
        # 1. Основные модули
        print("1. Импорт основных модулей...")
        import pandas as pd
        print("   OK pandas")
        
        # 2. Существующие модули
        print("2. Импорт существующих модулей...")
        from models.grid_strategy import GridStrategy
        print("   OK GridStrategy")
        from services.backtester import Backtester
        print("   OK Backtester")
        
        # 3. Индикаторы
        print("3. Импорт индикаторов...")
        from src.indicators.bollinger_bands import BollingerBands
        print("   OK BollingerBands")
        from src.indicators.fractals import FractalDetector
        print("   OK FractalDetector")
        
        # 4. Анализ рынка
        print("4. Импорт анализа рынка...")
        from src.market_analysis.trend_detector import TrendDetector
        print("   OK TrendDetector")
        from src.market_analysis.volatility import VolatilityCalculator
        print("   OK VolatilityCalculator")
        
        # 5. Автонастройка
        print("5. Импорт модулей автонастройки...")
        from src.auto_config.grid_bounds import GridBoundsCalculator
        print("   OK GridBoundsCalculator")
        from src.auto_config.grid_levels import GridLevelsCalculator
        print("   OK GridLevelsCalculator")
        from src.auto_config.position_size import PositionSizeCalculator
        print("   OK PositionSizeCalculator")
        
        # 6. Новые классы v2.0
        print("6. Импорт новых классов v2.0...")
        from src.grid_strategy_v2 import AdaptiveGridStrategy
        print("   OK AdaptiveGridStrategy")
        from src.backtester_v2 import AdaptiveBacktester
        print("   OK AdaptiveBacktester")
        
        print("\nOK Все импорты успешны!")
        
        # Проверка создания экземпляров
        print("\nПроверка создания экземпляров...")
        
        # Создаем тестовые данные
        import numpy as np
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'open': np.random.normal(50000, 1000, 100),
            'high': np.random.normal(50500, 1000, 100),
            'low': np.random.normal(49500, 1000, 100),
            'close': np.random.normal(50000, 1000, 100),
            'volume': np.random.normal(100, 10, 100)
        }, index=dates)
        
        # AdaptiveGridStrategy
        print("   Создание AdaptiveGridStrategy...")
        strategy = AdaptiveGridStrategy(deposit=10000.0)
        print(f"   OK AdaptiveGridStrategy создан: {strategy}")
        
        # AdaptiveBacktester
        print("   Создание AdaptiveBacktester...")
        backtester = AdaptiveBacktester(initial_balance=10000.0, use_auto_config=True)
        print(f"   OK AdaptiveBacktester создан: {backtester}")
        
        print("\nOK Все проверки пройдены успешно!")
        return True
        
    except ImportError as e:
        print(f"\nERROR Ошибка импорта: {e}")
        print(f"   Traceback: {sys.exc_info()}")
        return False
    except Exception as e:
        print(f"\nERROR Ошибка при проверке: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)