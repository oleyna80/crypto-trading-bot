"""
Клиент для работы с API биржи Bybit через библиотеку ccxt.

Предоставляет методы для получения исторических данных и симуляции исполнения ордеров.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import ccxt
import pandas as pd

from config.settings import config

logger = logging.getLogger(__name__)


class ExchangeAPI:
    """Обёртка над ccxt для работы с Bybit API."""

    def __init__(self, use_testnet: bool = True) -> None:
        """
        Инициализация клиента Bybit.

        Args:
            use_testnet: Использовать тестовую сеть (по умолчанию True).
        """
        self.use_testnet = use_testnet
        self.exchange = self._create_exchange()

    def _create_exchange(self) -> ccxt.bybit:
        """Создаёт и настраивает экземпляр ccxt.bybit."""
        exchange_config = {
            "apiKey": config.api_key,
            "secret": config.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # Perpetual futures
        }

        if self.use_testnet:
            exchange_config["urls"] = {"api": {"public": config.testnet_url}}
            logger.info("Используется тестовая сеть Bybit.")
        else:
            exchange_config["urls"] = {"api": {"public": config.mainnet_url}}
            logger.info("Используется основная сеть Bybit.")

        exchange = ccxt.bybit(exchange_config)
        exchange.load_markets()
        return exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List[float]]:
        """
        Получение исторических данных OHLCV.

        Args:
            symbol: Торговая пара (например, 'BTC/USDT').
            timeframe: Таймфрейм ('15m', '1h', '1d' и т.д.).
            since: Временная метка начала в миллисекундах.
            limit: Максимальное количество свечей.

        Returns:
            Список списков [timestamp, open, high, low, close, volume].

        Raises:
            ccxt.NetworkError: Проблемы с сетью.
            ccxt.ExchangeError: Ошибка биржи.
        """
        try:
            logger.info(
                f"Загрузка данных {symbol} {timeframe} с {since} (limit={limit})"
            )
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, since=since, limit=limit
            )
            logger.info(f"Получено {len(ohlcv)} свечей.")
            return ohlcv
        except ccxt.NetworkError as e:
            logger.error(f"Ошибка сети при загрузке данных: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Ошибка биржи при загрузке данных: {e}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            raise

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Загружает исторические данные за указанный период.

        Args:
            symbol: Торговая пара.
            timeframe: Таймфрейм.
            start_date: Начальная дата.
            end_date: Конечная дата.

        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume.
        """
        import time

        since = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        all_ohlcv = []
        current_since = since
        limit = 1000
        max_iterations = 100  # защита от бесконечного цикла
        iteration = 0

        logger.info(
            f"Начинаем загрузку данных {symbol} {timeframe} "
            f"с {start_date} по {end_date}"
        )

        while current_since < end_timestamp and iteration < max_iterations:
            iteration += 1
            try:
                ohlcv = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit,
                )
                # Остановка цикла при пустом ответе
                if not ohlcv:
                    logger.info(f"Итерация {iteration}: API вернул пустой ответ, загрузка завершена")
                    break

                all_ohlcv.extend(ohlcv)
                # Логирование прогресса
                logger.info(
                    f"Итерация {iteration}: загружено {len(ohlcv)} свечей, "
                    f"всего {len(all_ohlcv)} свечей"
                )

                # Следующая итерация начинается с timestamp последней свечи + 1 мс
                last_timestamp = ohlcv[-1][0]
                new_since = last_timestamp + 1

                # Защита от бесконечного цикла: если timestamp не увеличивается, выходим
                if new_since <= current_since:
                    logger.warning(f"Timestamp не увеличивается ({new_since} <= {current_since}), загрузка прервана")
                    break

                current_since = new_since

                # Если получено меньше лимита, проверяем, достигли ли конца запрошенного периода
                if len(ohlcv) < limit:
                    # Если текущий since уже достиг или превысил end_timestamp, выходим
                    if current_since >= end_timestamp:
                        logger.info("Достигнут конец запрошенного периода, загрузка завершена")
                        break
                    # Иначе продолжаем загрузку (возможно, API ограничивает количество свечей за запрос)
                    logger.debug("Получено меньше лимита, но продолжаем загрузку...")
                else:
                    # Rate limiting (не добавляем задержку после последней итерации)
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Ошибка на итерации {iteration}: {e}")
                break

        logger.info(f"Загрузка завершена. Всего загружено {len(all_ohlcv)} свечей")

        # Фильтрация аномальных свечей по цене закрытия
        filtered_ohlcv = []
        for candle in all_ohlcv:
            close_price = candle[4]  # индекс 4 - цена закрытия
            if 20000 <= close_price <= 200000:
                filtered_ohlcv.append(candle)
        
        removed_count = len(all_ohlcv) - len(filtered_ohlcv)
        if removed_count > 0:
            logger.warning(
                f"Отфильтровано {removed_count} свечей с аномальными ценами "
                f"(осталось {len(filtered_ohlcv)} свечей)"
            )
        else:
            logger.debug("Аномальные цены не обнаружены")
        
        logger.info(f"Количество корректных свечей после фильтрации: {len(filtered_ohlcv)}")

        df = pd.DataFrame(
            filtered_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def simulate_order_execution(
        self,
        price: float,
        side: str,
        amount: float,
        current_time: datetime,
    ) -> Dict[str, Any]:
        """
        Симуляция исполнения ордера (для backtesting).

        В реальном трейдинге здесь будет вызов create_order.
        В симуляции просто возвращаем информацию о сделке.

        Args:
            price: Цена исполнения.
            side: 'buy' или 'sell'.
            amount: Объём в базовой валюте.
            current_time: Время исполнения.

        Returns:
            Словарь с информацией о сделке.
        """
        # В симуляции предполагаем мгновенное исполнение по указанной цене
        trade = {
            "timestamp": current_time,
            "side": side,
            "price": price,
            "amount": amount,
            "cost": price * amount,
            "fee": 0.0006 * price * amount,  # 0.06% комиссия Bybit
            "executed": True,
        }
        logger.debug(f"Симуляция ордера: {trade}")
        return trade

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Возвращает информацию о торговой паре.

        Args:
            symbol: Торговая пара.

        Returns:
            Словарь с информацией о лимитах, точности и т.д.
        """
        market = self.exchange.market(symbol)
        return {
            "precision": market["precision"],
            "limits": market["limits"],
            "active": market["active"],
        }