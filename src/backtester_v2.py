"""
Adaptive Backtester v2.0 - расширенный бэктестер для Grid Trading Bot v2.0.

Класс AdaptiveBacktester наследуется от Backtester и добавляет поддержку
автоматической настройки стратегии через AdaptiveGridStrategy.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from services.backtester import Backtester
from src.grid_strategy_v2 import AdaptiveGridStrategy
from services.data_loader import DataLoader

logger = logging.getLogger(__name__)


class AdaptiveBacktester(Backtester):
    """Расширенный бэктестер для v2.0 с поддержкой автонастройки"""
    
    def __init__(
        self,
        initial_balance: float,
        exchange: Optional[Any] = None,
        use_auto_config: bool = True
    ) -> None:
        """
        Инициализация адаптивного бэктестера.
        
        Args:
            initial_balance: Начальный баланс в USDT
            exchange: Экземпляр ExchangeAPI (если None, создаётся новый)
            use_auto_config: Использовать автонастройку стратегии
        """
        super().__init__(initial_balance, exchange)
        self.use_auto_config = use_auto_config
        self.adaptive_strategy: Optional[AdaptiveGridStrategy] = None
        self.configuration_history: list = []
        
        logger.info(f"Инициализирован AdaptiveBacktester с автонастройкой: {use_auto_config}")
    
    def run_with_auto_config(
        self,
        daily_data: pd.DataFrame,
        intraday_data: Optional[pd.DataFrame] = None,
        fee_rate: float = 0.0006
    ) -> Dict[str, Any]:
        """
        Запуск бэктеста с автонастройкой стратегии.
        
        Процесс:
        1. Автонастройка AdaptiveGridStrategy на основе данных
        2. Запуск стандартного бэктеста с настроенной стратегией
        3. Возврат расширенных результатов с информацией о конфигурации
        
        Args:
            daily_data: DataFrame с дневными данными (для анализа тренда)
            intraday_data: DataFrame с внутридневными данными (для торговли, опционально)
            fee_rate: Комиссия за сделку (по умолчанию 0.06%)
            
        Returns:
            Словарь с результатами бэктеста, включая информацию о конфигурации
            
        Raises:
            ValueError: При ошибках в данных или настройке
        """
        logger.info("Запуск бэктеста с автонастройкой")
        
        # Если внутридневные данные не переданы, используем дневные
        if intraday_data is None:
            logger.warning("Внутридневные данные не переданы, используем дневные данные для торговли")
            intraday_data = daily_data.copy()
        
        try:
            # 1. Создание и автонастройка стратегии
            logger.info("Шаг 1: Создание и автонастройка AdaptiveGridStrategy")
            self.adaptive_strategy = AdaptiveGridStrategy(deposit=self.initial_balance)
            
            # Для автонастройки нужны данные разной периодичности
            # daily_data - для анализа тренда и волатильности
            # intraday_data (15m) - для точной настройки параметров
            config_params = self.adaptive_strategy.auto_configure(
                daily_data=daily_data,
                data_15m=intraday_data,
                balance=self.initial_balance
            )
            
            # Сохраняем конфигурацию в историю
            self.configuration_history.append({
                "timestamp": pd.Timestamp.now(),
                "params": config_params.copy()
            })
            
            logger.info(f"Стратегия настроена: режим={config_params.get('mode')}, "
                       f"уровней={config_params.get('num_levels')}, "
                       f"границы=[{config_params.get('lower_bound'):.2f}, "
                       f"{config_params.get('upper_bound'):.2f}]")
            
            # 2. Запуск стандартного бэктеста
            logger.info("Шаг 2: Запуск стандартного бэктеста")
            standard_metrics = super().run_backtest(
                data=intraday_data,
                strategy=self.adaptive_strategy,
                fee_rate=fee_rate
            )
            
            # 3. Расширение результатов информацией о конфигурации
            logger.info("Шаг 3: Формирование расширенных результатов")
            extended_metrics = self._extend_metrics(standard_metrics, config_params)
            
            logger.info(f"Бэктест с автонастройкой завершён. Сделок: {extended_metrics.get('total_trades', 0)}")
            return extended_metrics
            
        except Exception as e:
            logger.error(f"Ошибка при запуске бэктеста с автонастройкой: {e}")
            # Fallback: запуск с консервативной стратегией
            return self._run_fallback_backtest(daily_data, intraday_data, fee_rate)
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: Optional[Any] = None,
        fee_rate: float = 0.0006
    ) -> Dict[str, Any]:
        """
        Переопределение метода run_backtest с поддержкой автонастройки.
        
        Если use_auto_config=True и strategy=None, создаётся AdaptiveGridStrategy
        с автонастройкой. Иначе используется стандартное поведение.
        
        Args:
            data: DataFrame с колонками ['open', 'high', 'low', 'close', 'volume']
            strategy: Экземпляр GridStrategy (опционально)
            fee_rate: Комиссия за сделку
            
        Returns:
            Словарь с результатами симуляции
        """
        # Если включена автонастройка и стратегия не передана
        if self.use_auto_config and strategy is None:
            logger.info("Используем автонастройку (стратегия не передана)")
            
            # Для автонастройки нужны дневные данные, но у нас только торговые данные
            # Используем те же данные для упрощения (в реальном сценарии нужны разные таймфреймы)
            daily_data = self._resample_to_daily(data) if len(data) > 100 else data
            
            return self.run_with_auto_config(
                daily_data=daily_data,
                intraday_data=data,
                fee_rate=fee_rate
            )
        else:
            # Используем стандартное поведение
            logger.info("Используем стандартный бэктест (стратегия передана или автонастройка отключена)")
            return super().run_backtest(data, strategy, fee_rate)
    
    def _extend_metrics(
        self,
        standard_metrics: Dict[str, Any],
        config_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Расширяет стандартные метрики информацией о конфигурации.
        
        Args:
            standard_metrics: Стандартные метрики бэктеста
            config_params: Параметры конфигурации стратегии
            
        Returns:
            Расширенные метрики
        """
        extended = standard_metrics.copy()
        
        # Добавляем информацию о конфигурации
        extended["configuration"] = {
            "mode": config_params.get("mode"),
            "auto_configured": config_params.get("auto_configured", True),
            "volatility": config_params.get("volatility"),
            "grid_range_percent": config_params.get("grid_range_percent"),
            "num_levels": config_params.get("num_levels"),
            "upper_bound": config_params.get("upper_bound"),
            "lower_bound": config_params.get("lower_bound"),
            "amount_per_level": config_params.get("amount_per_level"),
            "grid_step": config_params.get("grid_step"),
            "total_investment": config_params.get("total_investment")
        }
        
        # Добавляем аналитические метрики
        if "final_equity" in extended and "initial_balance" in extended:
            pnl = extended["final_equity"] - extended["initial_balance"]
            investment = config_params.get("total_investment", 0)
            
            if investment > 0:
                # ROI на инвестированный капитал
                extended["roi_on_investment"] = (pnl / investment) * 100
            else:
                extended["roi_on_investment"] = 0.0
            
            # Эффективность использования капитала
            if extended["initial_balance"] > 0:
                capital_utilization = (investment / extended["initial_balance"]) * 100
                extended["capital_utilization_percent"] = capital_utilization
            else:
                extended["capital_utilization_percent"] = 0.0
        
        # Добавляем информацию о режиме
        extended["trading_mode"] = config_params.get("mode", "UNKNOWN")
        extended["use_auto_config"] = self.use_auto_config
        
        return extended
    
    def _run_fallback_backtest(
        self,
        daily_data: pd.DataFrame,
        intraday_data: pd.DataFrame,
        fee_rate: float
    ) -> Dict[str, Any]:
        """
        Запуск бэктеста с fallback стратегией при ошибке автонастройки.
        
        Args:
            daily_data: Дневные данные
            intraday_data: Внутридневные данные
            fee_rate: Комиссия
            
        Returns:
            Результаты бэктеста
        """
        logger.warning("Используем fallback бэктест")
        
        try:
            # Создаем консервативную стратегию
            current_price = intraday_data['close'].iloc[-1] if len(intraday_data) > 0 else 50000.0
            
            fallback_strategy = AdaptiveGridStrategy(
                upper_bound=current_price * 1.10,
                lower_bound=current_price * 0.90,
                num_levels=15,
                amount_per_level=0.001,
                deposit=self.initial_balance
            )
            
            # Запускаем стандартный бэктест
            metrics = super().run_backtest(
                data=intraday_data,
                strategy=fallback_strategy,
                fee_rate=fee_rate
            )
            
            # Добавляем информацию о fallback
            metrics["fallback_mode"] = True
            metrics["configuration"] = {
                "mode": "RANGE",
                "auto_configured": False,
                "fallback": True,
                "upper_bound": current_price * 1.10,
                "lower_bound": current_price * 0.90,
                "num_levels": 15,
                "amount_per_level": 0.001
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка fallback бэктеста: {e}")
            # Возвращаем пустые метрики
            return {
                "error": str(e),
                "fallback_mode": True,
                "configuration": {"error": "Failed to run backtest"}
            }
    
    def _resample_to_daily(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ресемплирует данные до дневного таймфрейма.
        
        Args:
            data: Исходные данные (любой таймфрейм)
            
        Returns:
            Дневные данные
        """
        if data.empty:
            return data
        
        # Проверяем, есть ли временной индекс
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Данные не имеют DatetimeIndex, невозможно ресемплировать")
            return data
        
        # Ресемплируем до дневного таймфрейма
        try:
            daily_data = data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            logger.info(f"Ресемплировано {len(data)} записей до {len(daily_data)} дневных свечей")
            return daily_data
            
        except Exception as e:
            logger.error(f"Ошибка ресемплинга: {e}")
            return data
    
    def get_configuration_history(self) -> list:
        """
        Возвращает историю конфигураций стратегии.
        
        Returns:
            Список конфигураций
        """
        return self.configuration_history.copy()
    
    def get_adaptive_strategy(self) -> Optional[AdaptiveGridStrategy]:
        """
        Возвращает текущую адаптивную стратегию.
        
        Returns:
            Экземпляр AdaptiveGridStrategy или None
        """
        return self.adaptive_strategy
    
    def compare_configurations(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Сравнивает две конфигурации стратегии.
        
        Args:
            config1: Первая конфигурация
            config2: Вторая конфигурация
            
        Returns:
            Словарь с сравнением
        """
        comparison = {}
        
        for key in set(config1.keys()) | set(config2.keys()):
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Для числовых значений считаем разницу в процентах
                if val1 != 0:
                    diff_percent = ((val2 - val1) / val1) * 100
                else:
                    diff_percent = 0.0
                comparison[key] = {
                    "value1": val1,
                    "value2": val2,
                    "difference": val2 - val1,
                    "difference_percent": diff_percent
                }
            else:
                # Для нечисловых значений просто показываем оба значения
                comparison[key] = {
                    "value1": val1,
                    "value2": val2,
                    "different": val1 != val2
                }
        
        return comparison