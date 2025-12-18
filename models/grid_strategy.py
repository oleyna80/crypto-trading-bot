"""
Реализация стратегии Grid Trading.

Создаёт сетку ордеров с фиксированными уровнями и проверяет исполнение
при изменении цены.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    """Уровень сетки."""

    price: float
    side: str  # 'buy' или 'sell'
    volume: float
    executed: bool = False
    execution_time: Optional[float] = None


class GridStrategy:
    """Стратегия Grid Trading."""

    def __init__(
        self,
        upper_bound: float,
        lower_bound: float,
        num_levels: int,
        amount_per_level: float,
        deposit: float,
    ) -> None:
        """
        Инициализация стратегии.

        Args:
            upper_bound: Верхняя граница сетки.
            lower_bound: Нижняя граница сетки.
            num_levels: Количество уровней сетки.
            amount_per_level: Объём на каждый уровень (в базовой валюте).
            deposit: Начальный депозит в USDT.
        """
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.num_levels = num_levels
        self.amount_per_level = amount_per_level
        self.deposit = deposit

        self.levels: List[GridLevel] = []
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Проверка корректности параметров."""
        if self.upper_bound <= self.lower_bound:
            raise ValueError(
                f"Верхняя граница ({self.upper_bound}) должна быть больше нижней ({self.lower_bound})."
            )
        if self.num_levels <= 0:
            raise ValueError("Количество уровней должно быть положительным.")
        if self.amount_per_level <= 0:
            raise ValueError("Объём на уровень должен быть положительным.")
        if self.deposit <= 0:
            raise ValueError("Депозит должен быть положительным.")

        # Проверка достаточности депозита для размещения всех ордеров на покупку
        total_buy_cost = self.amount_per_level * self.num_levels * self.lower_bound
        if total_buy_cost > self.deposit:
            deficit = total_buy_cost - self.deposit
            logger.warning(
                f"Общая стоимость покупок ({total_buy_cost:.2f} USDT) "
                f"превышает депозит ({self.deposit:.2f} USDT). "
                f"Дефицит: {deficit:.2f} USDT. "
                f"Рекомендуется: уменьшить количество уровней или увеличить депозит."
            )

    def calculate_levels(self, current_price: Optional[float] = None) -> List[GridLevel]:
        """
        Рассчитывает уровни сетки.

        Args:
            current_price: Текущая цена (опционально, используется для логирования).

        Returns:
            Список объектов GridLevel.
        """
        if current_price is not None:
            logger.info(
                f"Расчёт уровней сетки: границы [{self.lower_bound}, {self.upper_bound}], "
                f"уровней {self.num_levels}, текущая цена {current_price}"
            )

        step = (self.upper_bound - self.lower_bound) / (self.num_levels - 1)
        self.levels = []

        if current_price is None:
            # Если цена не передана, используем среднюю цену между границами
            current_price = (self.lower_bound + self.upper_bound) / 2
            logger.warning(
                f"current_price не передан, используется средняя цена {current_price:.2f}"
            )

        for i in range(self.num_levels):
            price = self.lower_bound + i * step
            # Уровни НИЖЕ текущей цены -> buy (покупаем дешевле)
            # Уровни ВЫШЕ текущей цены -> sell (продаём дороже)
            side = "buy" if price < current_price else "sell"

            level = GridLevel(price=price, side=side, volume=self.amount_per_level)
            self.levels.append(level)

        logger.info(f"Создано {len(self.levels)} уровней сетки.")
        return self.levels

    def check_order_execution(
        self,
        current_price: float,
        current_time: float,
        tolerance: float = 0.001,
    ) -> List[GridLevel]:
        """
        Проверяет, какие уровни сетки должны быть исполнены при текущей цене.

        Args:
            current_price: Текущая цена.
            current_time: Временная метка.
            tolerance: Допуск для сравнения цен (относительный).

        Returns:
            Список уровней, которые должны быть исполнены.
        """
        executed = []
        for level in self.levels:
            if level.executed:
                continue

            # Для buy ордера: если цена опустилась ниже или равна уровню
            # Для sell ордера: если цена поднялась выше или равна уровню
            if level.side == "buy" and current_price <= level.price * (1 + tolerance):
                level.executed = True
                level.execution_time = current_time
                executed.append(level)
                logger.debug(
                    f"Исполнение BUY ордера по цене {level.price:.2f} (текущая {current_price:.2f})"
                )
            elif level.side == "sell" and current_price >= level.price * (1 - tolerance):
                level.executed = True
                level.execution_time = current_time
                executed.append(level)
                logger.debug(
                    f"Исполнение SELL ордера по цене {level.price:.2f} (текущая {current_price:.2f})"
                )

        return executed

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по сетке."""
        executed = [l for l in self.levels if l.executed]
        buy_executed = [l for l in executed if l.side == "buy"]
        sell_executed = [l for l in executed if l.side == "sell"]

        total_buy_volume = sum(l.volume for l in buy_executed)
        total_sell_volume = sum(l.volume for l in sell_executed)

        return {
            "total_levels": len(self.levels),
            "executed_levels": len(executed),
            "buy_executed": len(buy_executed),
            "sell_executed": len(sell_executed),
            "total_buy_volume": total_buy_volume,
            "total_sell_volume": total_sell_volume,
            "remaining_levels": len(self.levels) - len(executed),
        }

    def reset(self) -> None:
        """Сброс состояния всех уровней (для повторного использования)."""
        for level in self.levels:
            level.executed = False
            level.execution_time = None
        logger.info("Состояние сетки сброшено.")