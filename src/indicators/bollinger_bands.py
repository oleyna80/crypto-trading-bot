"""
Модуль для расчета полос Боллинджера.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BollingerBands:
    """
    Класс для расчета полос Боллинджера.

    Полосы Боллинджера состоят из:
    - Средней линии (SMA) за период
    - Верхней и нижней полос на 1 и 2 стандартных отклонения
    - Ширины полос (bandwidth)

    Attributes:
        data (pd.Series): Временной ряд цен.
        period (int): Период для скользящего среднего (по умолчанию 24).
        num_std (float): Количество стандартных отклонений (по умолчанию 1.0).
    """

    def __init__(self, data: pd.Series, period: int = 24, num_std: float = 1.0):
        """
        Инициализация индикатора.

        Args:
            data: Ряд цен (например, закрытие).
            period: Период скользящего окна.
            num_std: Множитель стандартного отклонения.

        Raises:
            ValueError: Если данные пусты или период некорректен.
        """
        if data is None or len(data) == 0:
            raise ValueError("Данные не могут быть пустыми.")
        if period <= 1:
            raise ValueError("Период должен быть больше 1.")
        if num_std <= 0:
            raise ValueError("Количество стандартных отклонений должно быть положительным.")

        self.data = data
        self.period = period
        self.num_std = num_std

        logger.debug(f"Инициализирован BollingerBands с периодом {period} и std={num_std}")

    def calculate(self) -> Dict[str, pd.Series]:
        """
        Вычисляет полосы Боллинджера.

        Returns:
            Словарь с ключами:
            - 'middle': SMA (средняя линия)
            - 'upper_1sigma': верхняя полоса на 1 стандартное отклонение
            - 'lower_1sigma': нижняя полоса на 1 стандартное отклонение
            - 'upper_2sigma': верхняя полоса на 2 стандартных отклонения
            - 'lower_2sigma': нижняя полоса на 2 стандартных отклонения
            - 'bandwidth': относительная ширина полос (2 * std / middle)

        Raises:
            ValueError: Если данных недостаточно для расчета.
        """
        if len(self.data) < self.period:
            raise ValueError(
                f"Недостаточно данных для периода {self.period}. "
                f"Требуется минимум {self.period} точек, получено {len(self.data)}."
            )

        # Проверка на NaN
        if self.data.isna().any():
            logger.warning("В данных присутствуют NaN значения. Они будут проигнорированы в расчетах.")

        middle = self.data.rolling(window=self.period, min_periods=self.period).mean()
        std = self.data.rolling(window=self.period, min_periods=self.period).std()

        # Полосы на 1 и 2 стандартных отклонения
        upper_1sigma = middle + self.num_std * std
        lower_1sigma = middle - self.num_std * std
        upper_2sigma = middle + 2 * self.num_std * std
        lower_2sigma = middle - 2 * self.num_std * std

        # Ширина полос (в процентах от средней линии)
        bandwidth = (2 * std) / middle

        result = {
            'middle': middle,
            'upper_1sigma': upper_1sigma,
            'lower_1sigma': lower_1sigma,
            'upper_2sigma': upper_2sigma,
            'lower_2sigma': lower_2sigma,
            'bandwidth': bandwidth
        }

        logger.info(f"Рассчитаны полосы Боллинджера для {len(self.data)} точек.")
        return result

    def get_last_values(self) -> Optional[Dict[str, float]]:
        """
        Возвращает последние рассчитанные значения (для последней доступной точки).

        Returns:
            Словарь с последними значениями или None, если расчет невозможен.
        """
        try:
            bands = self.calculate()
            last = {}
            for key, series in bands.items():
                if not series.empty:
                    last[key] = series.iloc[-1]
                else:
                    last[key] = np.nan
            return last
        except Exception as e:
            logger.error(f"Ошибка при получении последних значений: {e}")
            return None