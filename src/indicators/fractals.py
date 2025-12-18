"""
Модуль для обнаружения фракталов на ценовых данных.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FractalDetector:
    """
    Класс для обнаружения фракталов (разворотных точек) на графике.

    Фрактал вверх: свеча, у которой high выше high двух свечей слева и двух свечей справа.
    Фрактал вниз: свеча, у которой low ниже low двух свечей слева и двух свечей справа.

    Attributes:
        data (pd.DataFrame): DataFrame с колонками 'high', 'low', 'close' (и опционально 'open', 'volume').
        lookback (int): Количество дней для подтверждения фрактала (по умолчанию 2).
    """

    def __init__(self, data: pd.DataFrame, lookback: int = 2):
        """
        Инициализация детектора фракталов.

        Args:
            data: DataFrame с колонками 'high' и 'low'.
            lookback: Количество свечей для подтверждения (по умолчанию 2).

        Raises:
            ValueError: Если данные не содержат необходимых колонок.
        """
        required_cols = {'high', 'low'}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Данные должны содержать колонки: {required_cols}")

        self.data = data.copy()
        self.lookback = lookback

        logger.debug(f"Инициализирован FractalDetector с lookback={lookback}")

    def find_fractals(self) -> Dict[str, List[Dict]]:
        """
        Находит все фракталы в данных.

        Returns:
            Словарь с ключами:
            - 'up_fractals': список словарей с информацией о фракталах вверх
            - 'down_fractals': список словарей с информацией о фракталах вниз
            - 'last_up': последний фрактал вверх или None
            - 'last_down': последний фрактал вниз или None

        Каждый фрактал представлен словарем:
            {
                'index': int,          # индекс в DataFrame
                'timestamp': pd.Timestamp,  # временная метка
                'price': float,        # цена (high для up, low для down)
                'confirmed': bool      # подтвержден ли фрактал
            }
        """
        highs = self.data['high'].values
        lows = self.data['low'].values
        indices = self.data.index

        up_fractals = []
        down_fractals = []

        # Минимальное количество свечей для проверки
        min_candles = self.lookback * 2 + 1
        if len(self.data) < min_candles:
            logger.warning(
                f"Недостаточно данных для обнаружения фракталов. "
                f"Требуется минимум {min_candles} свечей, получено {len(self.data)}."
            )
            return {
                'up_fractals': [],
                'down_fractals': [],
                'last_up': None,
                'last_down': None
            }

        # Поиск фракталов
        for i in range(self.lookback, len(self.data) - self.lookback):
            # Проверка фрактала вверх
            if self._is_up_fractal(i, highs):
                confirmed = self._confirm_fractal(i, direction='up')
                fractal_info = {
                    'index': i,
                    'timestamp': indices[i],
                    'price': highs[i],
                    'confirmed': confirmed
                }
                up_fractals.append(fractal_info)
                logger.debug(f"Найден фрактал вверх на индексе {i}, цена {highs[i]}, подтвержден: {confirmed}")

            # Проверка фрактала вниз
            if self._is_down_fractal(i, lows):
                confirmed = self._confirm_fractal(i, direction='down')
                fractal_info = {
                    'index': i,
                    'timestamp': indices[i],
                    'price': lows[i],
                    'confirmed': confirmed
                }
                down_fractals.append(fractal_info)
                logger.debug(f"Найден фрактал вниз на индексе {i}, цена {lows[i]}, подтвержден: {confirmed}")

        # Определяем последние фракталы
        last_up = up_fractals[-1] if up_fractals else None
        last_down = down_fractals[-1] if down_fractals else None

        logger.info(
            f"Обнаружено {len(up_fractals)} фракталов вверх и {len(down_fractals)} фракталов вниз."
        )

        return {
            'up_fractals': up_fractals,
            'down_fractals': down_fractals,
            'last_up': last_up,
            'last_down': last_down
        }

    def _is_up_fractal(self, idx: int, highs: np.ndarray) -> bool:
        """
        Проверяет, является ли свеча с индексом idx фракталом вверх.

        Условие: high[idx] > high[idx-2], high[idx-1], high[idx+1], high[idx+2]
        """
        left1 = highs[idx - 1] if idx - 1 >= 0 else -np.inf
        left2 = highs[idx - 2] if idx - 2 >= 0 else -np.inf
        right1 = highs[idx + 1] if idx + 1 < len(highs) else -np.inf
        right2 = highs[idx + 2] if idx + 2 < len(highs) else -np.inf

        current = highs[idx]
        return (current > left1 and current > left2 and
                current > right1 and current > right2)

    def _is_down_fractal(self, idx: int, lows: np.ndarray) -> bool:
        """
        Проверяет, является ли свеча с индексом idx фракталом вниз.

        Условие: low[idx] < low[idx-2], low[idx-1], low[idx+1], low[idx+2]
        """
        left1 = lows[idx - 1] if idx - 1 >= 0 else np.inf
        left2 = lows[idx - 2] if idx - 2 >= 0 else np.inf
        right1 = lows[idx + 1] if idx + 1 < len(lows) else np.inf
        right2 = lows[idx + 2] if idx + 2 < len(lows) else np.inf

        current = lows[idx]
        return (current < left1 and current < left2 and
                current < right1 and current < right2)

    def _confirm_fractal(self, idx: int, direction: str) -> bool:
        """
        Подтверждает фрактал через lookback дней.

        Для фрактала вверх: после idx цена не должна опускаться ниже high[idx] - порог.
        Для фрактала вниз: после idx цена не должна подниматься выше low[idx] + порог.

        Args:
            idx: Индекс фрактала.
            direction: 'up' или 'down'.

        Returns:
            True, если фрактал подтвержден.
        """
        if direction == 'up':
            price = self.data.iloc[idx]['high']
            # Проверяем следующие lookback свечей
            for i in range(1, self.lookback + 1):
                if idx + i >= len(self.data):
                    break
                if self.data.iloc[idx + i]['low'] < price * 0.99:  # Порог 1%
                    return False
            return True
        elif direction == 'down':
            price = self.data.iloc[idx]['low']
            for i in range(1, self.lookback + 1):
                if idx + i >= len(self.data):
                    break
                if self.data.iloc[idx + i]['high'] > price * 1.01:  # Порог 1%
                    return False
            return True
        else:
            raise ValueError(f"Неизвестное направление: {direction}")

    def get_fractal_zones(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Возвращает зоны фракталов (кластеризованные).

        Returns:
            Словарь с ключами 'up_zones' и 'down_zones', где каждый элемент
            является кортежем (индекс, цена).
        """
        fractals = self.find_fractals()
        up_zones = [(f['index'], f['price']) for f in fractals['up_fractals'] if f['confirmed']]
        down_zones = [(f['index'], f['price']) for f in fractals['down_fractals'] if f['confirmed']]

        return {
            'up_zones': up_zones,
            'down_zones': down_zones
        }