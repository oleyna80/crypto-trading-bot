"""
Модуль детектора тренда на основе полос Боллинджера и фракталов.
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any
from src.indicators.bollinger_bands import BollingerBands
from src.indicators.fractals import FractalDetector

logger = logging.getLogger(__name__)


class TrendDetector:
    """
    Детектор тренда, использующий полосы Боллинджера и фракталы.

    Логика определения тренда (согласно ТЗ):

    DOWNTREND:
    - Цена < нижняя полоса BB (-1σ) - ГЛАВНОЕ УСЛОВИЕ
    - Down Fractal пробит вниз
    - Цена НЕ вернулась выше фрактала

    UPTREND:
    - Цена > верхняя полоса BB (+1σ) - ГЛАВНОЕ УСЛОВИЕ
    - Up Fractal пробит вверх
    - Цена НЕ вернулась ниже фрактала

    RANGE:
    - Цена внутри полос ±1σ

    Attributes:
        daily_data (pd.DataFrame): Данные с колонками 'open', 'high', 'low', 'close', 'volume'.
        bb_period (int): Период для полос Боллинджера (по умолчанию 24).
        fractal_lookback (int): Lookback для подтверждения фракталов (по умолчанию 2).
    """

    def __init__(self, daily_data: pd.DataFrame, bb_period: int = 24, fractal_lookback: int = 2):
        """
        Инициализация детектора тренда.

        Args:
            daily_data: DataFrame с дневными данными.
            bb_period: Период для полос Боллинджера.
            fractal_lookback: Lookback для подтверждения фракталов.

        Raises:
            ValueError: Если данные не содержат необходимых колонок.
        """
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(daily_data.columns):
            raise ValueError(f"Данные должны содержать колонки: {required_cols}")

        self.daily_data = daily_data.copy()
        self.bb_period = bb_period
        self.fractal_lookback = fractal_lookback

        logger.debug(
            f"Инициализирован TrendDetector с bb_period={bb_period}, "
            f"fractal_lookback={fractal_lookback}"
        )

    def detect_trend(self) -> str:
        """
        Определяет текущий тренд на основе последних данных.

        Returns:
            "UPTREND" | "DOWNTREND" | "RANGE"

        Raises:
            ValueError: Если данных недостаточно для анализа.
        """
        if len(self.daily_data) < self.bb_period:
            logger.warning(
                f"Недостаточно данных для анализа тренда. "
                f"Требуется минимум {self.bb_period} точек, получено {len(self.daily_data)}. "
                f"Возвращаем RANGE."
            )
            return "RANGE"

        # Получаем последние значения
        latest_close = self.daily_data['close'].iloc[-1]
        latest_high = self.daily_data['high'].iloc[-1]
        latest_low = self.daily_data['low'].iloc[-1]

        # Рассчитываем полосы Боллинджера
        bb = BollingerBands(self.daily_data['close'], period=self.bb_period, num_std=1.0)
        try:
            bands = bb.calculate()
        except ValueError as e:
            logger.error(f"Ошибка расчета полос Боллинджера: {e}. Возвращаем RANGE.")
            return "RANGE"

        # Последние значения полос
        upper_1sigma = bands['upper_1sigma'].iloc[-1]
        lower_1sigma = bands['lower_1sigma'].iloc[-1]

        # Определяем положение цены относительно полос
        price_above_upper = latest_close > upper_1sigma
        price_below_lower = latest_close < lower_1sigma
        price_inside = not (price_above_upper or price_below_lower)

        logger.debug(
            f"Цена закрытия: {latest_close:.4f}, "
            f"Верхняя полоса BB: {upper_1sigma:.4f}, "
            f"Нижняя полоса BB: {lower_1sigma:.4f}"
        )

        # Если цена внутри полос - RANGE
        if price_inside:
            logger.info("Цена внутри полос ±1σ -> RANGE")
            return "RANGE"

        # Анализ фракталов
        fractal_detector = FractalDetector(self.daily_data, lookback=self.fractal_lookback)
        try:
            fractals = fractal_detector.find_fractals()
        except Exception as e:
            logger.error(f"Ошибка анализа фракталов: {e}. Используем только BB.")
            # Если фракталы недоступны, определяем тренд только по BB
            if price_above_upper:
                return "UPTREND"
            elif price_below_lower:
                return "DOWNTREND"
            else:
                return "RANGE"

        last_up = fractals['last_up']
        last_down = fractals['last_down']

        # UPTREND логика
        if price_above_upper:
            logger.debug("Цена выше верхней полосы BB (+1σ) - проверяем фракталы для UPTREND")
            if last_up and self._is_fractal_broken_up(last_up, latest_low):
                logger.info("UPTREND подтвержден: цена выше верхней полосы BB и пробит Up Fractal")
                return "UPTREND"
            else:
                logger.info("UPTREND (только по BB, фрактал не подтвержден)")
                return "UPTREND"

        # DOWNTREND логика
        elif price_below_lower:
            logger.debug("Цена ниже нижней полосы BB (-1σ) - проверяем фракталы для DOWNTREND")
            if last_down and self._is_fractal_broken_down(last_down, latest_high):
                logger.info("DOWNTREND подтвержден: цена ниже нижней полосы BB и пробит Down Fractal")
                return "DOWNTREND"
            else:
                logger.info("DOWNTREND (только по BB, фрактал не подтвержден)")
                return "DOWNTREND"

        # На всякий случай (должно быть обработано выше)
        return "RANGE"

    def _is_fractal_broken_up(self, up_fractal: Dict[str, Any], current_low: float) -> bool:
        """
        Проверяет, пробит ли Up Fractal вверх и цена не вернулась ниже.

        Args:
            up_fractal: Словарь с информацией о фрактале вверх.
            current_low: Текущее значение low.

        Returns:
            True, если фрактал пробит и цена не вернулась ниже.
        """
        fractal_price = up_fractal['price']
        # Фрактал пробит вверх, если текущий low выше цены фрактала
        broken = current_low > fractal_price
        if broken:
            logger.debug(f"Up Fractal пробит вверх: цена фрактала {fractal_price:.4f}, текущий low {current_low:.4f}")
        else:
            logger.debug(f"Up Fractal не пробит: цена фрактала {fractal_price:.4f}, текущий low {current_low:.4f}")
        return broken

    def _is_fractal_broken_down(self, down_fractal: Dict[str, Any], current_high: float) -> bool:
        """
        Проверяет, пробит ли Down Fractal вниз и цена не вернулась выше.

        Args:
            down_fractal: Словарь с информацией о фрактале вниз.
            current_high: Текущее значение high.

        Returns:
            True, если фрактал пробит и цена не вернулась выше.
        """
        fractal_price = down_fractal['price']
        # Фрактал пробит вниз, если текущий high ниже цены фрактала
        broken = current_high < fractal_price
        if broken:
            logger.debug(f"Down Fractal пробит вниз: цена фрактала {fractal_price:.4f}, текущий high {current_high:.4f}")
        else:
            logger.debug(f"Down Fractal не пробит: цена фрактала {fractal_price:.4f}, текущий high {current_high:.4f}")
        return broken

    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Возвращает детальный анализ тренда с метриками.

        Returns:
            Словарь с детальной информацией.
        """
        trend = self.detect_trend()
        latest_close = self.daily_data['close'].iloc[-1]

        # Рассчитываем полосы Боллинджера
        bb = BollingerBands(self.daily_data['close'], period=self.bb_period, num_std=1.0)
        bands = bb.calculate()
        upper_1sigma = bands['upper_1sigma'].iloc[-1]
        lower_1sigma = bands['lower_1sigma'].iloc[-1]

        # Анализ фракталов
        fractal_detector = FractalDetector(self.daily_data, lookback=self.fractal_lookback)
        fractals = fractal_detector.find_fractals()

        return {
            'trend': trend,
            'close_price': latest_close,
            'bb_upper': upper_1sigma,
            'bb_lower': lower_1sigma,
            'bb_middle': bands['middle'].iloc[-1],
            'price_relative_to_bb': (latest_close - bands['middle'].iloc[-1]) / bands['middle'].iloc[-1],
            'last_up_fractal': fractals['last_up'],
            'last_down_fractal': fractals['last_down'],
            'up_fractals_count': len(fractals['up_fractals']),
            'down_fractals_count': len(fractals['down_fractals']),
            'timestamp': self.daily_data.index[-1]
        }