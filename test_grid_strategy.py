"""
Модульное тестирование GridStrategy.
"""
import sys
sys.path.append('.')

from models.grid_strategy import GridStrategy, GridLevel
import logging

logging.basicConfig(level=logging.WARNING)

def test_grid_strategy_initialization():
    """Тест инициализации GridStrategy."""
    print("Тест 1: Инициализация GridStrategy")
    try:
        strategy = GridStrategy(
            upper_bound=50000.0,
            lower_bound=40000.0,
            num_levels=10,
            amount_per_level=0.001,
            deposit=1000.0
        )
        assert strategy.upper_bound == 50000.0
        assert strategy.lower_bound == 40000.0
        assert strategy.num_levels == 10
        assert strategy.amount_per_level == 0.001
        assert strategy.deposit == 1000.0
        print("  [OK] Успешно")
        return True
    except Exception as e:
        print(f"  [FAIL] Ошибка: {e}")
        return False

def test_grid_strategy_validation():
    """Тест валидации параметров."""
    print("Тест 2: Валидация параметров")
    # Неправильные параметры
    try:
        GridStrategy(upper_bound=40000, lower_bound=50000, num_levels=10, amount_per_level=0.001, deposit=1000)
        print("  [FAIL] Ожидалось исключение (верхняя граница меньше нижней)")
        return False
    except ValueError as e:
        print(f"  [OK] Исключение поймано: {e}")
    
    try:
        GridStrategy(upper_bound=50000, lower_bound=40000, num_levels=0, amount_per_level=0.001, deposit=1000)
        print("  [FAIL] Ожидалось исключение (num_levels <= 0)")
        return False
    except ValueError as e:
        print(f"  [OK] Исключение поймано: {e}")
    
    try:
        GridStrategy(upper_bound=50000, lower_bound=40000, num_levels=10, amount_per_level=-0.001, deposit=1000)
        print("  [FAIL] Ожидалось исключение (amount_per_level <= 0)")
        return False
    except ValueError as e:
        print(f"  [OK] Исключение поймано: {e}")
    
    try:
        GridStrategy(upper_bound=50000, lower_bound=40000, num_levels=10, amount_per_level=0.001, deposit=-1000)
        print("  [FAIL] Ожидалось исключение (deposit <= 0)")
        return False
    except ValueError as e:
        print(f"  [OK] Исключение поймано: {e}")
    
    return True

def test_calculate_levels():
    """Тест расчёта уровней сетки."""
    print("Тест 3: Расчёт уровней сетки")
    strategy = GridStrategy(
        upper_bound=50000.0,
        lower_bound=40000.0,
        num_levels=5,
        amount_per_level=0.001,
        deposit=1000.0
    )
    levels = strategy.calculate_levels(current_price=45000.0)
    assert len(levels) == 5
    # Проверка цен
    expected_prices = [40000.0, 42500.0, 45000.0, 47500.0, 50000.0]
    # Ожидаемые стороны на основе текущей цены 45000:
    # price < 45000 -> buy, price >= 45000 -> sell
    expected_sides = ["buy", "buy", "sell", "sell", "sell"]
    for i, level in enumerate(levels):
        assert abs(level.price - expected_prices[i]) < 0.01
        assert level.side == expected_sides[i], f"Уровень {level.price}: ожидалось {expected_sides[i]}, получено {level.side}"
        assert level.volume == 0.001
        assert not level.executed
    print("  [OK] Уровни рассчитаны корректно")
    return True

def test_check_order_execution():
    """Тест проверки исполнения ордеров."""
    print("Тест 4: Проверка исполнения ордеров")
    strategy = GridStrategy(
        upper_bound=50000.0,
        lower_bound=40000.0,
        num_levels=5,
        amount_per_level=0.001,
        deposit=1000.0
    )
    # Передаём текущую цену для определения сторон
    strategy.calculate_levels(current_price=45000.0)
    # Уровни при current_price=45000:
    # 40000 buy, 42500 buy, 45000 sell, 47500 sell, 50000 sell
    # При цене 39000 должны сработать buy уровни 40000 и 42500 (цена <= уровень)
    executed = strategy.check_order_execution(current_price=39000.0, current_time=1234567890)
    # Допуск tolerance=0.001, значит условие: 39000 <= level.price * 1.001
    # Для 40000: 40040 -> True
    # Для 42500: 42542.5 -> True
    # Для sell уровней условие не выполняется
    if len(executed) != 2:
        print(f"  [FAIL] Ожидалось 2 исполненных ордера, получено {len(executed)}")
        return False
    # Проверим, что исполнены именно buy 40000 и 42500
    found_40000 = False
    found_42500 = False
    for e in executed:
        if e.price == 40000.0 and e.side == "buy":
            found_40000 = True
            assert e.executed
        if e.price == 42500.0 and e.side == "buy":
            found_42500 = True
            assert e.executed
    if not (found_40000 and found_42500):
        print(f"  [FAIL] Исполненные ордера: {[(e.price, e.side) for e in executed]}")
        return False
    print("  [OK] Исполнение buy ордеров корректно")
    
    # Проверка sell ордера
    executed = strategy.check_order_execution(current_price=48000.0, current_time=1234567891)
    # Должен сработать sell уровень 47500 (цена >= уровень)
    # Уровень 50000 не сработает, потому что 48000 < 50000 * 0.999 = 49950
    found = False
    for e in executed:
        if e.price == 47500.0 and e.side == "sell":
            found = True
    if not found:
        print(f"  [FAIL] Исполненные ордера: {[(e.price, e.side) for e in executed]}")
        return False
    print("  [OK] Исполнение sell ордера корректно")
    return True

def test_statistics():
    """Тест статистики."""
    print("Тест 5: Статистика сетки")
    strategy = GridStrategy(
        upper_bound=50000.0,
        lower_bound=40000.0,
        num_levels=5,
        amount_per_level=0.001,
        deposit=1000.0
    )
    strategy.calculate_levels(current_price=45000.0)
    stats = strategy.get_statistics()
    assert stats["total_levels"] == 5
    assert stats["executed_levels"] == 0
    assert stats["buy_executed"] == 0
    assert stats["sell_executed"] == 0
    # Исполним два buy ордера (при цене 39000 сработают 40000 и 42500)
    strategy.check_order_execution(current_price=39000.0, current_time=1234567890)
    stats = strategy.get_statistics()
    assert stats["executed_levels"] == 2
    assert stats["buy_executed"] == 2
    assert stats["total_buy_volume"] == 0.002  # 0.001 * 2
    print("  [OK] Статистика корректна")
    return True

def test_reset():
    """Тест сброса состояния."""
    print("Тест 6: Сброс состояния")
    strategy = GridStrategy(
        upper_bound=50000.0,
        lower_bound=40000.0,
        num_levels=5,
        amount_per_level=0.001,
        deposit=1000.0
    )
    strategy.calculate_levels(current_price=45000.0)
    strategy.check_order_execution(current_price=39000.0, current_time=1234567890)
    stats = strategy.get_statistics()
    assert stats["executed_levels"] == 2  # два buy ордера
    strategy.reset()
    stats = strategy.get_statistics()
    assert stats["executed_levels"] == 0
    print("  [OK] Сброс выполнен")
    return True

def run_all_tests():
    """Запуск всех тестов."""
    print("=== Модульное тестирование GridStrategy ===")
    results = []
    results.append(test_grid_strategy_initialization())
    results.append(test_grid_strategy_validation())
    results.append(test_calculate_levels())
    results.append(test_check_order_execution())
    results.append(test_statistics())
    results.append(test_reset())
    
    passed = sum(results)
    total = len(results)
    print(f"\nИтог: {passed}/{total} тестов пройдено")
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)