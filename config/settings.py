"""
Конфигурация приложения Grid Trading Bot.

Загружает переменные окружения из .env файла и предоставляет
настройки для подключения к Bybit API и параметров стратегии.
"""

import os
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Конфигурация приложения."""

    # API ключи Bybit
    api_key: str = os.getenv("BYBIT_API_KEY", "")
    api_secret: str = os.getenv("BYBIT_API_SECRET", "")

    # URL API (testnet/mainnet)
    testnet_url: str = os.getenv("BYBIT_TESTNET_URL", "https://api-testnet.bybit.com")
    mainnet_url: str = os.getenv("BYBIT_MAINNET_URL", "https://api.bybit.com")

    # Торговые параметры
    symbol: str = os.getenv("SYMBOL", "BTC/USDT:USDT")
    timeframe: str = os.getenv("TIMEFRAME", "15m")
    initial_balance: float = float(os.getenv("INITIAL_BALANCE", "1000.0"))

    # Параметры сетки
    grid_levels: int = int(os.getenv("GRID_LEVELS", "20"))
    grid_step_pct: float = float(os.getenv("GRID_STEP_PCT", "0.5"))
    order_size: float = float(os.getenv("ORDER_SIZE", "0.001"))

    # Диапазон цен (опционально, можно вычислять динамически)
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None

    # Режим работы
    use_testnet: bool = True

    def validate(self) -> None:
        """Проверка корректности конфигурации."""
        # FIX (REVIEW.md): Валидация API ключей только для mainnet.
        # Для testnet ключи не обязательны, так как публичный API может работать без них.
        if not self.use_testnet and (not self.api_key or not self.api_secret):
            raise ValueError(
                "API ключи не заданы. Укажите BYBIT_API_KEY и BYBIT_API_SECRET в .env файле."
            )
        if self.grid_levels <= 0:
            raise ValueError("Количество уровней сетки должно быть положительным.")
        if self.grid_step_pct <= 0:
            raise ValueError("Шаг сетки (в процентах) должен быть положительным.")
        if self.order_size <= 0:
            raise ValueError("Размер ордера должен быть положительным.")
        if self.initial_balance <= 0:
            raise ValueError("Начальный баланс должен быть положительным.")

    @property
    def api_url(self) -> str:
        """Возвращает URL API в зависимости от режима."""
        return self.testnet_url if self.use_testnet else self.mainnet_url

    @property
    def exchange_id(self) -> str:
        """Идентификатор биржи для ccxt."""
        return "bybit"


# Глобальный экземпляр конфигурации
config = Config()