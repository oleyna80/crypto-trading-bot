"""
Загрузчик исторических данных с Bybit.

Предоставляет методы для получения и предобработки данных,
включая кэширование для ускорения повторных загрузок.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path

import pandas as pd

from models.exchange_api import ExchangeAPI
from config.settings import config

logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и управления историческими данными."""

    def __init__(self, cache_dir: str = "./cache") -> None:
        """
        Инициализация загрузчика.

        Args:
            cache_dir: Директория для кэширования данных.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.exchange = ExchangeAPI(use_testnet=config.use_testnet)

    def _get_cache_filename(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> Path:
        """Генерирует имя файла кэша на основе параметров."""
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        # Заменяем '/' и ':' на '_' в символе, чтобы избежать создания поддиректорий и недопустимых символов
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        filename = f"{safe_symbol}_{timeframe}_{start_str}_{end_str}.pkl"
        return self.cache_dir / filename

    def load_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Загружает исторические данные за указанный период.

        Args:
            symbol: Торговая пара (например, 'BTC/USDT').
            timeframe: Таймфрейм ('15m', '1h', и т.д.).
            start_date: Начальная дата.
            end_date: Конечная дата.
            use_cache: Использовать кэш, если данные уже загружены.

        Returns:
            DataFrame с колонками: open, high, low, close, volume.
        """
        logger.info(
            f"Загрузка данных {symbol} {timeframe} с {start_date} по {end_date}"
        )

        # Проверка кэша
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        if use_cache and cache_file.exists():
            logger.info(f"Загрузка из кэша: {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    df = pickle.load(f)
                logger.info(f"Загружено {len(df)} строк из кэша.")
                return df
            except Exception as e:
                logger.warning(f"Ошибка загрузки кэша: {e}. Загружаем заново.")

        # Загрузка с биржи
        try:
            df = self.exchange.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки данных с биржи: {e}")
            # Возвращаем пустой DataFrame
            df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            df.index.name = "timestamp"

        # Сохранение в кэш (только если данные не пустые)
        if use_cache and not df.empty:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(df, f)
                logger.info(f"Данные сохранены в кэш: {cache_file}")
            except Exception as e:
                logger.warning(f"Не удалось сохранить кэш: {e}")

        return df

    def load_last_year_data(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Загружает данные за последний год (по умолчанию из конфига).

        Args:
            symbol: Торговая пара (по умолчанию из конфига).
            timeframe: Таймфрейм (по умолчанию из конфига).
            use_cache: Использовать кэш.

        Returns:
            DataFrame за последний год.
        """
        symbol = symbol or config.symbol
        timeframe = timeframe or config.timeframe

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        return self.load_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
        )

    def preprocess_data(
        self,
        df: pd.DataFrame,
        fill_missing: bool = True,
        resample: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Предобработка данных: заполнение пропусков, ресемплирование и т.д.

        Args:
            df: Исходный DataFrame.
            fill_missing: Заполнять пропущенные значения (forward fill).
            resample: Ресемплировать данные (например, '1H').

        Returns:
            Обработанный DataFrame.
        """
        df = df.copy()

        # Проверка на пропуски во времени
        if df.index.is_monotonic_increasing:
            expected_freq = pd.infer_freq(df.index)
            if expected_freq is None:
                logger.warning("Не удалось определить частоту данных.")
            else:
                # Создание полного временного ряда
                full_range = pd.date_range(start=df.index[0], end=df.index[-1], freq=expected_freq)
                df = df.reindex(full_range)
                if fill_missing:
                    df = df.ffill()

        # Ресемплирование (если требуется)
        if resample:
            df = df.resample(resample).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            df = df.dropna()

        logger.info(f"Данные предобработаны. Форма: {df.shape}")
        return df

    def get_data_statistics(self, df: pd.DataFrame) -> dict:
        """
        Возвращает статистику по данным.

        Args:
            df: DataFrame с данными.

        Returns:
            Словарь со статистикой.
        """
        if df.empty:
            return {}

        stats = {
            "start_date": df.index[0],
            "end_date": df.index[-1],
            "total_rows": len(df),
            "columns": list(df.columns),
            "price_range": {
                "min": df["low"].min(),
                "max": df["high"].max(),
                "mean": df["close"].mean(),
                "median": df["close"].median(),
            },
            "volume_stats": {
                "total": df["volume"].sum(),
                "average": df["volume"].mean(),
            },
            "missing_values": df.isnull().sum().to_dict(),
            "median_price": df["close"].median(),
            "mean_price": df["close"].mean(),
        }
        return stats