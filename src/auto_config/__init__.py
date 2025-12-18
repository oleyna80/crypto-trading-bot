"""
Модуль автонастройки параметров сетки для Grid Trading Bot v2.0.

Содержит калькуляторы для автоматического расчёта:
- Границ сетки (GridBoundsCalculator)
- Количества уровней и шага (GridLevelsCalculator)
- Размера позиции (PositionSizeCalculator)
"""

from .grid_bounds import GridBoundsCalculator
from .grid_levels import GridLevelsCalculator
from .position_size import PositionSizeCalculator

__all__ = [
    "GridBoundsCalculator",
    "GridLevelsCalculator",
    "PositionSizeCalculator",
]