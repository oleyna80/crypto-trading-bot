"""
Движок бэктестинга для Grid Trading стратегии.

Проводит симуляцию торговли на исторических данных,
собирает сделки и рассчитывает метрики производительности.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from models.grid_strategy import GridStrategy
from models.exchange_api import ExchangeAPI
from config.settings import config

logger = logging.getLogger(__name__)


class Backtester:
    """Класс для проведения бэктестинга."""

    def __init__(
        self,
        initial_balance: float,
        exchange: Optional[ExchangeAPI] = None,
    ) -> None:
        """
        Инициализация бэктестера.

        Args:
            initial_balance: Начальный баланс в USDT.
            exchange: Экземпляр ExchangeAPI (если None, создаётся новый).
        """
        self.initial_balance = initial_balance
        self.exchange = exchange or ExchangeAPI(use_testnet=config.use_testnet)
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: GridStrategy,
        fee_rate: float = 0.0006,
    ) -> Dict[str, Any]:
        """
        Запускает симуляцию торговли на исторических данных.

        Args:
            data: DataFrame с колонками ['open', 'high', 'low', 'close', 'volume'].
            strategy: Экземпляр GridStrategy.
            fee_rate: Комиссия за сделку (по умолчанию 0.06%).

        Returns:
            Словарь с результатами симуляции.
        """
        logger.info(
            f"Запуск бэктеста на {len(data)} свечах, начальный баланс: {self.initial_balance} USDT"
        )

        # Сброс состояния
        self.trades = []
        self.equity_curve = []
        strategy.reset()

        # Инициализация переменных
        balance = self.initial_balance
        position = 0.0  # Количество BTC
        equity = balance

        # Предварительный расчёт уровней сетки на основе медианной цены (устойчивой к выбросам)
        avg_price = data["close"].median()
        logger.info(f"Медианная цена данных: {avg_price:.2f}")
        strategy.calculate_levels(current_price=avg_price)

        # Счётчики для группирования warnings
        insufficient_funds_count = 0
        insufficient_position_count = 0
        
        # Основной цикл по свечам
        for idx, row in data.iterrows():
            current_price = row["close"]
            current_time = idx
            # Если idx - это Timestamp, используем .timestamp()
            if hasattr(current_time, 'timestamp'):
                timestamp = current_time.timestamp()
            else:
                timestamp = float(current_time)

            # Проверка исполнения ордеров
            executed_levels = strategy.check_order_execution(
                current_price=current_price,
                current_time=timestamp,
            )

            for level in executed_levels:
                # Симуляция исполнения ордера
                trade = self.exchange.simulate_order_execution(
                    price=level.price,
                    side=level.side,
                    amount=level.volume,
                    current_time=current_time,
                )

                # Обновление баланса и позиции
                if level.side == "buy":
                    cost = level.price * level.volume
                    fee = cost * fee_rate
                    if balance >= cost + fee:
                        balance -= cost + fee
                        position += level.volume
                        trade["balance_change"] = -(cost + fee)
                    else:
                        insufficient_funds_count += 1
                        continue
                else:  # sell
                    if position >= level.volume:
                        revenue = level.price * level.volume
                        fee = revenue * fee_rate
                        balance += revenue - fee
                        position -= level.volume
                        trade["balance_change"] = revenue - fee
                    else:
                        insufficient_position_count += 1
                        continue

                # Сохранение сделки
                trade.update(
                    {
                        "balance_after": balance,
                        "position_after": position,
                        "level_price": level.price,
                    }
                )
                self.trades.append(trade)

            # Расчёт текущей equity (баланс + стоимость позиции)
            equity = balance + position * current_price
            self.equity_curve.append(
                {
                    "timestamp": current_time,
                    "equity": equity,
                    "balance": balance,
                    "position": position,
                    "price": current_price,
                }
            )
        
        # Вывод итоговых warnings
        if insufficient_funds_count > 0:
            logger.warning(
                f"Пропущено {insufficient_funds_count} покупок из-за недостатка средств"
            )
        if insufficient_position_count > 0:
            logger.warning(
                f"Пропущено {insufficient_position_count} продаж из-за отсутствия позиции"
            )

        # Финальные метрики
        metrics = self.calculate_metrics()

        logger.info(
            f"Бэктест завершён. Сделок: {len(self.trades)}, Финальная equity: {equity:.2f} USDT"
        )
        return metrics

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Рассчитывает метрики производительности на основе сделок и equity curve.

        Returns:
            Словарь с метриками.
        """
        if not self.trades:
            logger.warning("Нет сделок для расчёта метрик.")
            return {}

        # Преобразование equity curve в DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if equity_df.empty:
            return {}

        # Общий PnL
        initial_equity = self.initial_balance
        final_equity = equity_df["equity"].iloc[-1]
        total_pnl = final_equity - initial_equity
        total_pnl_pct = (total_pnl / initial_equity) * 100

        # FIX (REVIEW.md): Правильный расчёт Win Rate на основе сопоставления покупок и продаж (FIFO)
        trades_df = pd.DataFrame(self.trades)
        win_rate = 0.0
        if not trades_df.empty and "side" in trades_df.columns and "price" in trades_df.columns:
            # Сортируем сделки по времени
            trades_df = trades_df.sort_values("timestamp")
            buy_stack = []  # список цен покупок
            win_count = 0
            total_pairs = 0
            for _, trade in trades_df.iterrows():
                if trade["side"] == "buy":
                    buy_stack.append(trade["price"])
                elif trade["side"] == "sell" and buy_stack:
                    buy_price = buy_stack.pop(0)  # FIFO
                    total_pairs += 1
                    if trade["price"] > buy_price:
                        win_count += 1
            if total_pairs > 0:
                win_rate = win_count / total_pairs

        # Максимальная просадка (Max Drawdown)
        # Убедимся, что timestamp является DatetimeIndex
        equity_df_copy = equity_df.copy()
        if not isinstance(equity_df_copy["timestamp"].iloc[0], pd.Timestamp):
            # Если timestamp не Timestamp, преобразуем в datetime
            equity_df_copy["timestamp"] = pd.to_datetime(equity_df_copy["timestamp"])
        equity_series = equity_df_copy.set_index("timestamp")["equity"]
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # FIX (REVIEW.md): Исправлен расчёт Sharpe Ratio с использованием annualization factor sqrt(252)
        # для торговых дней (предполагаем дневные returns после ресемплинга)
        # Ресемплируем только если есть достаточно данных
        if len(equity_series) > 1:
            try:
                equity_series_resampled = equity_series.resample("D").last().ffill()
                if len(equity_series_resampled) > 1:
                    returns = equity_series_resampled.pct_change().dropna()
                    if len(returns) > 1 and returns.std() > 0:
                        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
                    else:
                        sharpe_ratio = 0.0
                else:
                    sharpe_ratio = 0.0
            except Exception:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Подсчёт buy и sell сделок
        buy_trades = 0
        sell_trades = 0
        if not trades_df.empty and 'side' in trades_df.columns:
            buy_trades = len(trades_df[trades_df['side'] == 'buy'])
            sell_trades = len(trades_df[trades_df['side'] == 'sell'])
        
        # Сборка результата
        metrics = {
            "initial_balance": self.initial_balance,
            "final_equity": final_equity,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(self.trades),
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "avg_trade_size": trades_df["amount"].mean() if not trades_df.empty else 0,
        }
        return metrics

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Возвращает сделки в виде DataFrame."""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

    def get_equity_dataframe(self) -> pd.DataFrame:
        """Возвращает equity curve в виде DataFrame."""
        return pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()

    def visualize_equity(self, save_path: Optional[str] = None) -> Any:
        """
        Создаёт график equity curve с помощью plotly.

        Args:
            save_path: Путь для сохранения графика (опционально).

        Returns:
            Объект plotly Figure.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly не установлен. Установите: pip install plotly")
            return None

        equity_df = self.get_equity_dataframe()
        if equity_df.empty:
            logger.warning("Нет данных для визуализации.")
            return None

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Equity Curve", "Drawdown"),
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_df["timestamp"],
                y=equity_df["equity"],
                mode="lines",
                name="Equity",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # Drawdown
        equity_series = equity_df.set_index("timestamp")["equity"]
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        fig.add_trace(
            go.Scatter(
                x=equity_series.index,
                y=drawdown,
                mode="lines",
                name="Drawdown %",
                line=dict(color="red"),
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title="Backtest Results",
            xaxis_title="Time",
            height=600,
            showlegend=True,
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"График сохранён в {save_path}")

        return fig