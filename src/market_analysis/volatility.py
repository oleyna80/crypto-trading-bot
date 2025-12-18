"""
Модуль калькулятора волатильности для Grid Trading Bot v2.0.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Union

logger = logging.getLogger(__name__)


class VolatilityCalculator:
    """
    Калькулятор волатильности с различными методами расчета.

    Предоставляет методы для:
    - Исторической волатильности (стандартное отклонение доходностей)
    - Average True Range (ATR)
    - Волатильности на основе полос Боллинджера
    - Относительной волатильности (по сравнению со средним значением)
    """

    @staticmethod
    def calculate_historical_volatility(data: pd.Series, period: int = 30) -> float:
        """
        Рассчитывает историческую волатильность как стандартное отклонение доходностей.

        Args:
            data: Временной ряд цен (например, закрытие).
            period: Период для расчета (количество последних точек).

        Returns:
            Историческая волатильность (в десятичной форме).

        Raises:
            ValueError: Если данных недостаточно или период некорректен.
        """
        if data is None or len(data) == 0:
            raise ValueError("Данные не могут быть пустыми.")
        if period <= 1:
            raise ValueError("Период должен быть больше 1.")
        if len(data) < period:
            logger.warning(
                f"Недостаточно данных для расчета исторической волатильности. "
                f"Требуется минимум {period} точек, получено {len(data)}. "
                f"Используем все доступные данные."
            )
            period = len(data)

        # Рассчитываем логарифмические доходности
        returns = np.log(data / data.shift(1)).dropna()

        if len(returns) < 2:
            logger.warning("Недостаточно данных для расчета доходностей. Возвращаем 0.")
            return 0.0

        # Берем последние period доходностей
        recent_returns = returns.tail(period)

        # Стандартное отклонение доходностей (ежедневная волатильность)
        volatility = recent_returns.std()

        logger.debug(
            f"Рассчитана историческая волатильность за период {period}: {volatility:.6f}"
        )

        return float(volatility)

    @staticmethod
    def calculate_annualized_volatility(data: pd.Series, period: int = 30, trading_days: int = 365) -> float:
        """
        Рассчитывает годовую волатильность.

        Args:
            data: Временной ряд цен.
            period: Период для расчета ежедневной волатильности.
            trading_days: Количество торговых дней в году.

        Returns:
            Годовая волатильность (в десятичной форме).
        """
        daily_vol = VolatilityCalculator.calculate_historical_volatility(data, period)
        annualized_vol = daily_vol * np.sqrt(trading_days)
        logger.debug(f"Годовая волатильность: {annualized_vol:.6f}")
        return float(annualized_vol)

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
        """
        Рассчитывает Average True Range (ATR).

        Args:
            data: DataFrame с колонками 'high', 'low', 'close'.
            period: Период для скользящего среднего.

        Returns:
            Значение ATR.

        Raises:
            ValueError: Если данные не содержат необходимых колонок.
        """
        required_cols = {'high', 'low', 'close'}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Данные должны содержать колонки: {required_cols}")

        if len(data) < period:
            logger.warning(
                f"Недостаточно данных для расчета ATR. "
                f"Требуется минимум {period} точек, получено {len(data)}. "
                f"Используем все доступные данные."
            )
            period = len(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Average True Range (простое скользящее среднее)
        atr = true_range.rolling(window=period, min_periods=1).mean().iloc[-1]

        logger.debug(f"Рассчитан ATR за период {period}: {atr:.6f}")

        return float(atr)

    @staticmethod
    def calculate_bb_volatility(data: pd.Series, period: int = 24, num_std: float = 1.0) -> Dict[str, float]:
        """
        Рассчитывает волатильность на основе ширины полос Боллинджера.

        Args:
            data: Временной ряд цен.
            period: Период для полос Боллинджера.
            num_std: Количество стандартных отклонений.

        Returns:
            Словарь с метриками волатильности:
            - 'bandwidth': относительная ширина полос (2 * std / middle)
            - 'std': стандартное отклонение
            - 'middle': средняя линия (SMA)
        """
        if len(data) < period:
            raise ValueError(
                f"Недостаточно данных для расчета полос Боллинджера. "
                f"Требуется минимум {period} точек, получено {len(data)}."
            )

        # Рассчитываем полосы Боллинджера вручную
        middle = data.rolling(window=period, min_periods=period).mean().iloc[-1]
        std = data.rolling(window=period, min_periods=period).std().iloc[-1]

        # Относительная ширина полос
        bandwidth = (2 * num_std * std) / middle if middle != 0 else 0.0

        result = {
            'bandwidth': float(bandwidth),
            'std': float(std),
            'middle': float(middle),
            'upper': float(middle + num_std * std),
            'lower': float(middle - num_std * std)
        }

        logger.debug(
            f"Волатильность по полосам Боллинджера: bandwidth={bandwidth:.6f}, std={std:.6f}"
        )

        return result

    @staticmethod
    def calculate_relative_volatility(data: pd.Series, benchmark: pd.Series, period: int = 30) -> float:
        """
        Рассчитывает относительную волатильность по сравнению с бенчмарком.

        Args:
            data: Временной ряд цен актива.
            benchmark: Временной ряд цен бенчмарка.
            period: Период для расчета.

        Returns:
            Отношение волатильности актива к волатильности бенчмарка.
        """
        vol_asset = VolatilityCalculator.calculate_historical_volatility(data, period)
        vol_benchmark = VolatilityCalculator.calculate_historical_volatility(benchmark, period)

        if vol_benchmark == 0:
            logger.warning("Волатильность бенчмарка равна 0. Возвращаем 0.")
            return 0.0

        relative_vol = vol_asset / vol_benchmark

        logger.debug(f"Относительная волатильность: {relative_vol:.4f}")

        return float(relative_vol)

    @staticmethod
    def get_volatility_regime(data: pd.Series, period: int = 30, threshold_high: float = 0.02,
                              threshold_low: float = 0.005) -> str:
        """
        Определяет режим волатильности.

        Args:
            data: Временной ряд цен.
            period: Период для расчета.
            threshold_high: Порог для высокой волатильности.
            threshold_low: Порог для низкой волатильности.

        Returns:
            "HIGH", "LOW" или "NORMAL"
        """
        volatility = VolatilityCalculator.calculate_historical_volatility(data, period)

        if volatility > threshold_high:
            regime = "HIGH"
        elif volatility < threshold_low:
            regime = "LOW"
        else:
            regime = "NORMAL"

        logger.info(f"Режим волатильности: {regime} (волатильность={volatility:.6f})")

        return regime

    @staticmethod
    def calculate_multiple_metrics(data: pd.DataFrame, price_col: str = 'close',
                                   period: int = 30) -> Dict[str, float]:
        """
        Рассчитывает несколько метрик волатильности одновременно.

        Args:
            data: DataFrame с ценовыми данными.
            price_col: Название колонки с ценами.
            period: Базовый период для расчетов.

        Returns:
            Словарь с метриками волатильности.
        """
        if price_col not in data.columns:
            raise ValueError(f"Колонка {price_col} не найдена в данных.")

        price_series = data[price_col]

        metrics = {
            'historical_volatility': VolatilityCalculator.calculate_historical_volatility(price_series, period),
            'annualized_volatility': VolatilityCalculator.calculate_annualized_volatility(price_series, period),
            'bb_bandwidth': VolatilityCalculator.calculate_bb_volatility(price_series, period)['bandwidth'],
            'volatility_regime': VolatilityCalculator.get_volatility_regime(price_series, period)
        }

        # Добавляем ATR, если есть необходимые колонки
        if {'high', 'low', 'close'}.issubset(data.columns):
            metrics['atr'] = VolatilityCalculator.calculate_atr(data, period=14)

        logger.info(f"Рассчитаны метрики волатильности: {metrics}")

        return metrics