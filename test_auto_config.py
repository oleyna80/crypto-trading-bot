#!/usr/bin/env python3
"""
Тестовый скрипт для проверки модулей автонастройки.
"""

import sys
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Добавляем путь к src
sys.path.insert(0, 'src')

from auto_config import GridBoundsCalculator, GridLevelsCalculator, PositionSizeCalculator


def test_grid_bounds_calculator():
    """Тест GridBoundsCalculator."""
    print("\n=== Тест GridBoundsCalculator ===")
    
    # Тестовые данные
    bb_data = {
        "upper": 65000.0,  # Верхняя граница Bollinger Bands (2σ)
        "middle": 60000.0, # Средняя линия (SMA)
        "lower": 55000.0   # Нижняя граница Bollinger Bands (2σ)
    }
    current_price = 61000.0
    volatility = 0.02  # 2% волатильность
    
    try:
        # Тестируем разные режимы
        for mode in ["RANGE", "UPTREND", "DOWNTREND"]:
            upper, lower = GridBoundsCalculator.calculate_bounds(
                mode, bb_data, current_price, volatility
            )
            print(f"{mode}: upper={upper:.2f}, lower={lower:.2f}, range={upper-lower:.2f}")
        
        # Тест с некорректными данными
        print("\nТест с некорректными данными:")
        try:
            GridBoundsCalculator.calculate_bounds("INVALID", bb_data, current_price, volatility)
        except ValueError as e:
            print(f"Ожидаемая ошибка: {e}")
            
    except Exception as e:
        print(f"Ошибка в тесте GridBoundsCalculator: {e}")
        return False
    
    return True


def test_grid_levels_calculator():
    """Тест GridLevelsCalculator."""
    print("\n=== Тест GridLevelsCalculator ===")
    
    try:
        # Тестируем разные режимы
        for mode in ["RANGE", "UPTREND", "DOWNTREND"]:
            levels = GridLevelsCalculator.calculate_optimal_levels(
                mode=mode,
                volatility=0.02,
                balance=5000.0,
                grid_range=10000.0
            )
            print(f"{mode}: optimal_levels={levels}")
        
        # Тест расчёта шага
        upper = 65000.0
        lower = 55000.0
        num_levels = 25
        step = GridLevelsCalculator.calculate_grid_step(upper, lower, num_levels)
        print(f"\nШаг сетки: upper={upper}, lower={lower}, levels={num_levels}, step={step:.2f}")
        
        # Тест с некорректными данными
        print("\nТест с некорректными данными:")
        try:
            GridLevelsCalculator.calculate_grid_step(55000.0, 65000.0, 10)  # upper < lower
        except ValueError as e:
            print(f"Ожидаемая ошибка: {e}")
            
    except Exception as e:
        print(f"Ошибка в тесте GridLevelsCalculator: {e}")
        return False
    
    return True


def test_position_size_calculator():
    """Тест PositionSizeCalculator."""
    print("\n=== Тест PositionSizeCalculator ===")
    
    try:
        # Тестируем разные режимы
        for mode in ["RANGE", "UPTREND", "DOWNTREND"]:
            order_size = PositionSizeCalculator.calculate_order_size(
                balance=5000.0,
                num_levels=25,
                mode=mode,
                current_price=61000.0
            )
            print(f"{mode}: order_size={order_size:.6f} BTC (${order_size * 61000.0:.2f})")
        
        # Тест с малым балансом
        print("\nТест с малым балансом (5 USD):")
        order_size = PositionSizeCalculator.calculate_order_size(
            balance=5.0,
            num_levels=10,
            mode="RANGE",
            current_price=61000.0
        )
        print(f"order_size={order_size:.6f} BTC")
        
        # Тест с некорректными данными
        print("\nТест с некорректными данными:")
        try:
            PositionSizeCalculator.calculate_order_size(
                balance=-1000.0,
                num_levels=10,
                mode="RANGE",
                current_price=61000.0
            )
        except ValueError as e:
            print(f"Ожидаемая ошибка: {e}")
            
    except Exception as e:
        print(f"Ошибка в тесте PositionSizeCalculator: {e}")
        return False
    
    return True


def main():
    """Основная функция тестирования."""
    print("Тестирование модулей автонастройки Grid Trading Bot v2.0")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    # Запуск тестов
    if test_grid_bounds_calculator():
        tests_passed += 1
    
    if test_grid_levels_calculator():
        tests_passed += 1
    
    if test_position_size_calculator():
        tests_passed += 1
    
    # Итоги
    print("\n" + "=" * 60)
    print(f"Итоги: {tests_passed}/{tests_total} тестов пройдено успешно")
    
    if tests_passed == tests_total:
        print("✅ Все модули автонастройки работают корректно!")
        return 0
    else:
        print("❌ Некоторые тесты не прошли")
        return 1


if __name__ == "__main__":
    sys.exit(main())