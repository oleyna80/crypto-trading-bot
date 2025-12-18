"""
Grid Bounds Calculator - автоматический расчёт границ сетки.

Класс GridBoundsCalculator предоставляет статические методы для расчёта
верхней и нижней границ сетки на основе Bollinger Bands, текущей цены и волатильности.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class GridBoundsCalculator:
    """Калькулятор границ сетки."""
    
    # Минимальный диапазон сетки в процентах от текущей цены
    MIN_RANGE_PERCENT = 5.0
    
    @staticmethod
    def calculate_bounds(
        mode: str,
        bb_data: dict,
        current_price: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Автоматический расчёт upper_bound и lower_bound.
        
        Логика:
        - RANGE: Границы вокруг средней (±1.5σ = 86% цен)
        - UPTREND: Сетка сдвинута вверх (upper_bound = upper_2sigma * 1.2, lower_bound = middle)
        - DOWNTREND: Сетка сдвинута вниз (upper_bound = middle, lower_bound = lower_2sigma * 0.8)
        
        Проверка минимального диапазона (5%)
        
        Args:
            mode: Режим работы ("RANGE", "UPTREND", "DOWNTREND")
            bb_data: Словарь с данными Bollinger Bands:
                - "upper": верхняя граница (2σ)
                - "middle": средняя линия (SMA)
                - "lower": нижняя граница (2σ)
            current_price: Текущая цена актива
            volatility: Волатильность (стандартное отклонение)
            
        Returns:
            Кортеж (upper_bound, lower_bound)
            
        Raises:
            ValueError: При некорректных входных данных
        """
        # Валидация входных данных
        if mode not in ["RANGE", "UPTREND", "DOWNTREND"]:
            raise ValueError(f"Некорректный режим: {mode}. Допустимые значения: RANGE, UPTREND, DOWNTREND")
        
        required_keys = ["upper", "middle", "lower"]
        if not all(key in bb_data for key in required_keys):
            raise ValueError(f"bb_data должен содержать ключи: {required_keys}")
        
        upper_2sigma = bb_data["upper"]
        middle = bb_data["middle"]
        lower_2sigma = bb_data["lower"]
        
        if any(val <= 0 for val in [upper_2sigma, middle, lower_2sigma, current_price]):
            raise ValueError("Все цены должны быть положительными")
        
        logger.info(
            f"Расчёт границ сетки: mode={mode}, current_price={current_price:.2f}, "
            f"volatility={volatility:.4f}, BB: upper={upper_2sigma:.2f}, "
            f"middle={middle:.2f}, lower={lower_2sigma:.2f}"
        )
        
        # Расчёт границ в зависимости от режима
        if mode == "RANGE":
            # Границы вокруг средней (±1.5σ = 86% цен)
            sigma_1_5 = 1.5 * volatility
            upper_bound = middle + sigma_1_5
            lower_bound = middle - sigma_1_5
            
        elif mode == "UPTREND":
            # Сетка сдвинута вверх
            upper_bound = upper_2sigma * 1.2
            lower_bound = middle
            
        elif mode == "DOWNTREND":
            # Сетка сдвинута вниз
            upper_bound = middle
            lower_bound = lower_2sigma * 0.8
            
        else:
            # Fallback на случай, если режим не распознан (не должно случиться)
            upper_bound = upper_2sigma
            lower_bound = lower_2sigma
        
        # Проверка минимального диапазона (5% от текущей цены)
        min_range = current_price * (GridBoundsCalculator.MIN_RANGE_PERCENT / 100.0)
        actual_range = upper_bound - lower_bound
        
        if actual_range < min_range:
            logger.warning(
                f"Диапазон сетки слишком мал: {actual_range:.2f} < {min_range:.2f}. "
                f"Расширяем до минимального диапазона."
            )
            # Расширяем диапазон симметрично относительно средней точки
            midpoint = (upper_bound + lower_bound) / 2
            upper_bound = midpoint + min_range / 2
            lower_bound = midpoint - min_range / 2
        
        # Проверка, что верхняя граница выше нижней
        if upper_bound <= lower_bound:
            logger.error(
                f"Верхняя граница ({upper_bound:.2f}) <= нижней ({lower_bound:.2f}). "
                f"Используем fallback значения."
            )
            # Fallback: используем Bollinger Bands с небольшим запасом
            upper_bound = upper_2sigma * 1.05
            lower_bound = lower_2sigma * 0.95
        
        # Ограничение границ разумными значениями (не менее 50% и не более 200% от текущей цены)
        upper_bound = max(current_price * 0.5, min(upper_bound, current_price * 2.0))
        lower_bound = max(current_price * 0.5, min(lower_bound, current_price * 2.0))
        
        logger.info(
            f"Рассчитанные границы: upper={upper_bound:.2f}, lower={lower_bound:.2f}, "
            f"диапазон={upper_bound - lower_bound:.2f} ({((upper_bound - lower_bound) / current_price * 100):.1f}%)"
        )
        
        return upper_bound, lower_bound