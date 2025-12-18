"""
Модуль анализа рынка для Grid Trading Bot v2.0.

Содержит:
- TrendDetector: детектор тренда на основе полос Боллинджера и фракталов
- VolatilityCalculator: калькулятор волатильности (историческая, ATR)
"""

from .trend_detector import TrendDetector
from .volatility import VolatilityCalculator

__all__ = ["TrendDetector", "VolatilityCalculator"]