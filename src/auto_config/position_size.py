"""
Position Size Calculator - автоматический расчёт размера позиции.

Класс PositionSizeCalculator предоставляет статические методы для расчёта
размера ордера в BTC на основе баланса, количества уровней и режима работы.
"""

import logging

logger = logging.getLogger(__name__)


class PositionSizeCalculator:
    """Калькулятор размера позиции."""
    
    # Распределение капитала по режимам (доля от баланса)
    CAPITAL_ALLOCATION = {
        "RANGE": 0.7,    # 70% баланса
        "UPTREND": 0.4,  # 40% баланса
        "DOWNTREND": 0.4 # 40% баланса
    }
    
    # Минимальный и максимальный размер ордера в BTC
    MIN_ORDER_SIZE_BTC = 0.0001
    MAX_ORDER_SIZE_BTC = 0.01
    
    # Минимальный баланс для торговли (USD)
    MIN_BALANCE_USD = 10.0
    
    @staticmethod
    def calculate_order_size(
        balance: float,
        num_levels: int,
        mode: str,
        current_price: float
    ) -> float:
        """
        Автоматический расчёт размера ордера (BTC).
        
        Логика:
        1. Распределение капитала: RANGE=70%, TREND=40%
        2. Половина уровней - buy (предполагаем симметричную сетку)
        3. Конвертация в BTC по текущей цене
        4. Ограничения: 0.0001-0.01 BTC
        
        Args:
            balance: Доступный баланс (USD)
            num_levels: Количество уровней сетки
            mode: Режим работы ("RANGE", "UPTREND", "DOWNTREND")
            current_price: Текущая цена BTC/USD
            
        Returns:
            Размер ордера в BTC
            
        Raises:
            ValueError: При некорректных входных данных
        """
        # Валидация входных данных
        if mode not in ["RANGE", "UPTREND", "DOWNTREND"]:
            raise ValueError(f"Некорректный режим: {mode}. Допустимые значения: RANGE, UPTREND, DOWNTREND")
        
        if balance <= 0:
            raise ValueError(f"Баланс должен быть положительным: {balance}")
        
        if num_levels < 1:
            raise ValueError(f"Количество уровней должно быть >= 1: {num_levels}")
        
        if current_price <= 0:
            raise ValueError(f"Текущая цена должна быть положительной: {current_price}")
        
        logger.info(
            f"Расчёт размера ордера: balance={balance:.2f} USD, levels={num_levels}, "
            f"mode={mode}, price={current_price:.2f} USD/BTC"
        )
        
        # Проверка минимального баланса
        if balance < PositionSizeCalculator.MIN_BALANCE_USD:
            logger.warning(
                f"Баланс ({balance:.2f} USD) меньше минимального "
                f"({PositionSizeCalculator.MIN_BALANCE_USD} USD). "
                f"Используем минимальный размер ордера."
            )
            return PositionSizeCalculator.MIN_ORDER_SIZE_BTC
        
        # 1. Определяем долю капитала для сетки
        allocation = PositionSizeCalculator.CAPITAL_ALLOCATION.get(mode, 0.5)
        allocated_capital = balance * allocation
        logger.debug(f"Выделенный капитал: {allocated_capital:.2f} USD (allocation={allocation})")
        
        # 2. Учитываем, что только половина уровней - buy ордера
        # (в симметричной сетке половина уровней выше текущей цены - sell,
        # половина ниже - buy)
        buy_levels = num_levels // 2
        if buy_levels < 1:
            buy_levels = 1
        
        # Капитал на один buy ордер
        capital_per_buy = allocated_capital / buy_levels
        logger.debug(
            f"Капитал на buy ордер: {capital_per_buy:.2f} USD "
            f"(buy_levels={buy_levels})"
        )
        
        # 3. Конвертация в BTC
        order_size_btc = capital_per_buy / current_price
        
        # 4. Применяем ограничения
        order_size_btc = max(
            PositionSizeCalculator.MIN_ORDER_SIZE_BTC,
            min(order_size_btc, PositionSizeCalculator.MAX_ORDER_SIZE_BTC)
        )
        
        # 5. Дополнительная проверка: если размер слишком мал относительно цены,
        # увеличиваем до минимального разумного значения
        min_reasonable_btc = 0.0005  # 0.0005 BTC ~ $30 при цене $60k
        if order_size_btc < min_reasonable_btc:
            # Пытаемся увеличить, уменьшив количество buy уровней
            if buy_levels > 1:
                new_buy_levels = max(1, buy_levels // 2)
                capital_per_buy = allocated_capital / new_buy_levels
                order_size_btc = capital_per_buy / current_price
                logger.debug(
                    f"Увеличиваем размер ордера за счёт уменьшения buy уровней: "
                    f"{buy_levels} -> {new_buy_levels}, size={order_size_btc:.6f} BTC"
                )
        
        # Финальное применение ограничений
        order_size_btc = max(
            PositionSizeCalculator.MIN_ORDER_SIZE_BTC,
            min(order_size_btc, PositionSizeCalculator.MAX_ORDER_SIZE_BTC)
        )
        
        # Проверка, что размер ордера не превышает доступный капитал
        required_capital = order_size_btc * current_price * buy_levels
        if required_capital > allocated_capital * 1.1:  # 10% запас
            logger.warning(
                f"Требуемый капитал ({required_capital:.2f} USD) превышает выделенный "
                f"({allocated_capital:.2f} USD). Уменьшаем размер ордера."
            )
            # Уменьшаем пропорционально
            reduction_factor = allocated_capital / required_capital
            order_size_btc *= reduction_factor
        
        # Округление до 8 знаков (типичная точность BTC)
        order_size_btc = round(order_size_btc, 8)
        
        logger.info(
            f"Рассчитанный размер ордера: {order_size_btc:.6f} BTC "
            f"({order_size_btc * current_price:.2f} USD), "
            f"buy_levels={buy_levels}, allocation={allocation}"
        )
        
        return order_size_btc