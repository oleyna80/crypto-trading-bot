"""
Adaptive Grid Strategy v2.0 - расширение GridStrategy с автонастройкой.

Класс AdaptiveGridStrategy наследуется от GridStrategy и добавляет
автоматическую настройку параметров сетки на основе анализа рынка.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from models.grid_strategy import GridStrategy
from src.indicators.bollinger_bands import BollingerBands
from src.indicators.fractals import FractalDetector
from src.market_analysis.trend_detector import TrendDetector
from src.market_analysis.volatility import VolatilityCalculator
from src.auto_config.grid_bounds import GridBoundsCalculator
from src.auto_config.grid_levels import GridLevelsCalculator
from src.auto_config.position_size import PositionSizeCalculator

logger = logging.getLogger(__name__)


class AdaptiveGridStrategy(GridStrategy):
    """Расширение GridStrategy с автонастройкой"""
    
    def __init__(
        self,
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
        num_levels: Optional[int] = None,
        amount_per_level: Optional[float] = None,
        deposit: Optional[float] = None,
    ) -> None:
        """
        Инициализация адаптивной стратегии.
        
        Args:
            upper_bound: Верхняя граница сетки (опционально, будет настроена автоматически)
            lower_bound: Нижняя граница сетки (опционально, будет настроена автоматически)
            num_levels: Количество уровней сетки (опционально, будет настроено автоматически)
            amount_per_level: Объём на каждый уровень (опционально, будет настроен автоматически)
            deposit: Начальный депозит в USDT (обязательно для автонастройки)
        """
        # Если все параметры переданы, используем обычную инициализацию
        if all(param is not None for param in [upper_bound, lower_bound, num_levels, amount_per_level, deposit]):
            super().__init__(upper_bound, lower_bound, num_levels, amount_per_level, deposit)
        else:
            # Инициализация с заглушками, которые будут настроены позже
            super().__init__(1.0, 0.0, 1, 0.001, deposit or 1000.0)
            
        self.mode = "RANGE"
        self.auto_configured = False
        self.volatility = 0.0
        self.current_price = 0.0
        self.config_params: Dict[str, Any] = {}
        
        logger.info("Инициализирован AdaptiveGridStrategy")
    
    def auto_configure(
        self,
        daily_data: pd.DataFrame,
        data_15m: pd.DataFrame,
        balance: float
    ) -> Dict[str, Any]:
        """
        Полная автонастройка всех параметров сетки.
        
        Процесс настройки:
        1. Детекция режима (TrendDetector)
        2. BB данные (BollingerBands)
        3. Волатильность (VolatilityCalculator)
        4. Границы (GridBoundsCalculator)
        5. Уровни (GridLevelsCalculator)
        6. Шаг (GridLevelsCalculator)
        7. Размер ордера (PositionSizeCalculator)
        
        Args:
            daily_data: DataFrame с дневными данными (для анализа тренда)
            data_15m: DataFrame с 15-минутными данными (для точной настройки)
            balance: Доступный баланс в USDT
            
        Returns:
            dict с параметрами конфигурации
            
        Raises:
            ValueError: При ошибках в данных или расчетах
        """
        logger.info("Запуск автонастройки AdaptiveGridStrategy")
        
        try:
            # 1. Детекция режима (TrendDetector)
            logger.info("Шаг 1: Детекция режима рынка")
            trend_detector = TrendDetector(daily_data)
            self.mode = trend_detector.detect_trend()
            logger.info(f"Определён режим: {self.mode}")
            
            # 2. BB данные (BollingerBands)
            logger.info("Шаг 2: Расчет полос Боллинджера")
            bb = BollingerBands(daily_data['close'], period=24, num_std=2.0)
            bb_data = bb.calculate()
            
            # Получаем последние значения BB
            bb_last = {
                "upper": bb_data['upper_2sigma'].iloc[-1],
                "middle": bb_data['middle'].iloc[-1],
                "lower": bb_data['lower_2sigma'].iloc[-1]
            }
            
            # 3. Волатильность (VolatilityCalculator)
            logger.info("Шаг 3: Расчет волатильности")
            self.volatility = VolatilityCalculator.calculate_historical_volatility(
                daily_data['close'], period=30
            )
            logger.info(f"Рассчитана волатильность: {self.volatility:.6f}")
            
            # Текущая цена (последнее закрытие из 15-минутных данных для точности)
            self.current_price = data_15m['close'].iloc[-1]
            logger.info(f"Текущая цена: {self.current_price:.2f}")
            
            # 4. Границы (GridBoundsCalculator)
            logger.info("Шаг 4: Расчет границ сетки")
            upper_bound, lower_bound = GridBoundsCalculator.calculate_bounds(
                mode=self.mode,
                bb_data=bb_last,
                current_price=self.current_price,
                volatility=self.volatility
            )
            
            # 5. Уровни (GridLevelsCalculator)
            logger.info("Шаг 5: Расчет количества уровней")
            grid_range = upper_bound - lower_bound
            num_levels = GridLevelsCalculator.calculate_optimal_levels(
                mode=self.mode,
                volatility=self.volatility,
                balance=balance,
                grid_range=grid_range
            )
            
            # 6. Шаг (GridLevelsCalculator)
            logger.info("Шаг 6: Расчет шага сетки")
            grid_step = GridLevelsCalculator.calculate_grid_step(
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                num_levels=num_levels
            )
            
            # 7. Размер ордера (PositionSizeCalculator)
            logger.info("Шаг 7: Расчет размера ордера")
            amount_per_level = PositionSizeCalculator.calculate_order_size(
                balance=balance,
                num_levels=num_levels,
                mode=self.mode,
                current_price=self.current_price
            )
            
            # Обновляем параметры стратегии
            self.upper_bound = upper_bound
            self.lower_bound = lower_bound
            self.num_levels = num_levels
            self.amount_per_level = amount_per_level
            self.deposit = balance
            
            # Валидация параметров
            self._validate_parameters()
            
            # Сохраняем конфигурационные параметры
            self.config_params = {
                "mode": self.mode,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "num_levels": num_levels,
                "amount_per_level": amount_per_level,
                "grid_step": grid_step,
                "volatility": self.volatility,
                "current_price": self.current_price,
                "grid_range": grid_range,
                "grid_range_percent": (grid_range / self.current_price * 100) if self.current_price > 0 else 0,
                "total_investment": amount_per_level * num_levels * ((upper_bound + lower_bound) / 2),
                "balance": balance
            }
            
            self.auto_configured = True
            
            logger.info("Автонастройка успешно завершена")
            logger.info(f"Параметры сетки: границы [{lower_bound:.2f}, {upper_bound:.2f}], "
                       f"уровней {num_levels}, шаг {grid_step:.4f}, "
                       f"размер ордера {amount_per_level:.6f} BTC")
            
            return self.config_params
            
        except Exception as e:
            logger.error(f"Ошибка автонастройки: {e}")
            # Fallback: используем консервативные параметры
            return self._fallback_configuration(daily_data, balance)
    
    def _fallback_configuration(
        self,
        daily_data: pd.DataFrame,
        balance: float
    ) -> Dict[str, Any]:
        """
        Fallback конфигурация при ошибке автонастройки.
        
        Args:
            daily_data: DataFrame с данными
            balance: Доступный баланс
            
        Returns:
            dict с консервативными параметрами
        """
        logger.warning("Используем fallback конфигурацию")
        
        # Базовые параметры
        current_price = daily_data['close'].iloc[-1] if len(daily_data) > 0 else 50000.0
        self.current_price = current_price
        
        # Консервативные границы (±10%)
        upper_bound = current_price * 1.10
        lower_bound = current_price * 0.90
        
        # Консервативное количество уровней
        num_levels = 15
        
        # Консервативный размер ордера (1% баланса на уровень)
        amount_per_level = (balance * 0.01) / current_price
        amount_per_level = max(0.0001, min(amount_per_level, 0.01))
        
        # Обновляем параметры
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.num_levels = num_levels
        self.amount_per_level = amount_per_level
        self.deposit = balance
        self.mode = "RANGE"
        self.auto_configured = False
        
        grid_range = upper_bound - lower_bound
        grid_step = grid_range / (num_levels - 1)
        
        self.config_params = {
            "mode": "RANGE",
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "num_levels": num_levels,
            "amount_per_level": amount_per_level,
            "grid_step": grid_step,
            "volatility": 0.02,  # Консервативная волатильность
            "current_price": current_price,
            "grid_range": grid_range,
            "grid_range_percent": 20.0,  # ±10% = 20% диапазон
            "total_investment": amount_per_level * num_levels * ((upper_bound + lower_bound) / 2),
            "balance": balance,
            "fallback": True
        }
        
        logger.info(f"Fallback конфигурация: границы [{lower_bound:.2f}, {upper_bound:.2f}], "
                   f"уровней {num_levels}, размер ордера {amount_per_level:.6f} BTC")
        
        return self.config_params
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Возвращает текущую конфигурацию стратегии.
        
        Returns:
            dict с параметрами конфигурации
        """
        return self.config_params.copy()
    
    def is_auto_configured(self) -> bool:
        """
        Проверяет, была ли выполнена автонастройка.
        
        Returns:
            True если автонастройка выполнена
        """
        return self.auto_configured
    
    def calculate_levels(self, current_price: Optional[float] = None) -> list:
        """
        Переопределение метода calculate_levels с дополнительной логикой.
        
        Args:
            current_price: Текущая цена (опционально)
            
        Returns:
            Список уровней сетки
        """
        # Если автонастройка не выполнена и параметры не валидны, используем fallback
        if not self.auto_configured and (self.upper_bound <= self.lower_bound or self.num_levels <= 0):
            logger.warning("Параметры стратегии не настроены. Используем calculate_levels суперкласса с текущей ценой.")
            
        # Используем текущую цену из конфигурации, если не передана
        if current_price is None and hasattr(self, 'current_price') and self.current_price > 0:
            current_price = self.current_price
            
        return super().calculate_levels(current_price)
    
    def __str__(self) -> str:
        """Строковое представление стратегии."""
        base_str = super().__str__() if hasattr(super(), '__str__') else ""
        return (f"AdaptiveGridStrategy(mode={self.mode}, "
                f"auto_configured={self.auto_configured}, "
                f"volatility={self.volatility:.4f}, "
                f"current_price={self.current_price:.2f})")