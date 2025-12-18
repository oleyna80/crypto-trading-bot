"""
Grid Levels Calculator - автоматический расчёт количества уровней и шага сетки.

Класс GridLevelsCalculator предоставляет статические методы для расчёта
оптимального количества уровней сетки и шага между ордерами.
"""

import logging
import math

logger = logging.getLogger(__name__)


class GridLevelsCalculator:
    """Калькулятор уровней и шага сетки."""
    
    # Базовое количество уровней для каждого режима
    BASE_LEVELS = {
        "RANGE": 45,
        "UPTREND": 20,
        "DOWNTREND": 20
    }
    
    # Минимальное и максимальное количество уровней
    MIN_LEVELS = 10
    MAX_LEVELS = 60
    
    # Минимальный баланс для использования максимального количества уровней
    MIN_BALANCE_FOR_MAX_LEVELS = 1000.0  # USD
    
    @staticmethod
    def calculate_optimal_levels(
        mode: str,
        volatility: float,
        balance: float,
        grid_range: float
    ) -> int:
        """
        Автоматический расчёт количества уровней.
        
        Логика:
        1. Базовое количество: RANGE=45, TREND=20
        2. Адаптация к волатильности: чем выше волатильность, тем меньше уровней
        3. Проверка капитала: при малом балансе уменьшаем количество уровней
        4. Ограничения: 10-60 уровней
        
        Args:
            mode: Режим работы ("RANGE", "UPTREND", "DOWNTREND")
            volatility: Волатильность (стандартное отклонение)
            balance: Доступный баланс (USD)
            grid_range: Диапазон сетки (upper_bound - lower_bound)
            
        Returns:
            Оптимальное количество уровней
            
        Raises:
            ValueError: При некорректных входных данных
        """
        # Валидация входных данных
        if mode not in ["RANGE", "UPTREND", "DOWNTREND"]:
            raise ValueError(f"Некорректный режим: {mode}. Допустимые значения: RANGE, UPTREND, DOWNTREND")
        
        if volatility <= 0:
            raise ValueError(f"Волатильность должна быть положительной: {volatility}")
        
        if balance <= 0:
            raise ValueError(f"Баланс должен быть положительным: {balance}")
        
        if grid_range <= 0:
            raise ValueError(f"Диапазон сетки должен быть положительным: {grid_range}")
        
        logger.info(
            f"Расчёт оптимального количества уровней: mode={mode}, "
            f"volatility={volatility:.4f}, balance={balance:.2f} USD, "
            f"grid_range={grid_range:.2f}"
        )
        
        # 1. Базовое количество уровней
        base_levels = GridLevelsCalculator.BASE_LEVELS.get(mode, 20)
        logger.debug(f"Базовое количество уровней: {base_levels}")
        
        # 2. Адаптация к волатильности
        # Нормализуем волатильность (предполагаем, что типичная волатильность ~0.02)
        typical_volatility = 0.02
        volatility_factor = typical_volatility / max(volatility, 0.001)
        
        # Ограничиваем коэффициент адаптации (0.5 - 2.0)
        volatility_factor = max(0.5, min(volatility_factor, 2.0))
        
        adjusted_by_volatility = int(base_levels * volatility_factor)
        logger.debug(f"С учётом волатильности: {adjusted_by_volatility} (factor={volatility_factor:.2f})")
        
        # 3. Проверка капитала
        # При малом балансе уменьшаем количество уровней
        if balance < GridLevelsCalculator.MIN_BALANCE_FOR_MAX_LEVELS:
            balance_factor = balance / GridLevelsCalculator.MIN_BALANCE_FOR_MAX_LEVELS
            balance_factor = max(0.3, balance_factor)  # Минимум 30% от базового
            adjusted_by_balance = int(adjusted_by_volatility * balance_factor)
            logger.debug(
                f"С учётом баланса: {adjusted_by_balance} "
                f"(balance_factor={balance_factor:.2f})"
            )
        else:
            adjusted_by_balance = adjusted_by_volatility
        
        # 4. Ограничения: 10-60 уровней
        optimal_levels = max(
            GridLevelsCalculator.MIN_LEVELS,
            min(adjusted_by_balance, GridLevelsCalculator.MAX_LEVELS)
        )
        
        # 5. Дополнительная проверка: для маленького диапазона уменьшаем уровни
        # Если диапазон меньше 2% от средней цены, уменьшаем уровни
        # (предполагаем среднюю цену ~ grid_range * 2 для оценки)
        estimated_price = grid_range * 2
        range_percent = (grid_range / estimated_price) * 100 if estimated_price > 0 else 0
        
        if range_percent < 2.0:
            reduction_factor = max(0.5, range_percent / 2.0)
            optimal_levels = int(optimal_levels * reduction_factor)
            logger.debug(
                f"Уменьшение из-за малого диапазона: {optimal_levels} "
                f"(range_percent={range_percent:.1f}%, factor={reduction_factor:.2f})"
            )
        
        # Финальная проверка ограничений
        optimal_levels = max(
            GridLevelsCalculator.MIN_LEVELS,
            min(optimal_levels, GridLevelsCalculator.MAX_LEVELS)
        )
        
        # Убедимся, что количество уровней нечётное для симметричной сетки
        if optimal_levels % 2 == 0:
            optimal_levels += 1
            logger.debug(f"Сделано нечётным: {optimal_levels}")
        
        logger.info(
            f"Оптимальное количество уровней: {optimal_levels} "
            f"(базовое: {base_levels}, волатильность: {volatility:.4f})"
        )
        
        return optimal_levels
    
    @staticmethod
    def calculate_grid_step(upper_bound: float, lower_bound: float, num_levels: int) -> float:
        """
        Расчёт шага между ордерами.
        
        Args:
            upper_bound: Верхняя граница сетки
            lower_bound: Нижняя граница сетки
            num_levels: Количество уровней
            
        Returns:
            Шаг между уровнями (в единицах цены)
            
        Raises:
            ValueError: При некорректных входных данных
        """
        # Валидация входных данных
        if upper_bound <= lower_bound:
            raise ValueError(
                f"Верхняя граница ({upper_bound}) должна быть больше нижней ({lower_bound})"
            )
        
        if num_levels < 2:
            raise ValueError(f"Количество уровней должно быть >= 2: {num_levels}")
        
        # Расчёт шага
        grid_range = upper_bound - lower_bound
        step = grid_range / (num_levels - 1)
        
        logger.info(
            f"Расчёт шага сетки: upper={upper_bound:.2f}, lower={lower_bound:.2f}, "
            f"levels={num_levels}, step={step:.4f}"
        )
        
        return step