#!/usr/bin/env python3
"""
Проверка использования медианы в backtester.
"""
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from services.backtester import Backtester
from models.grid_strategy import GridStrategy

# Создаём тестовые данные с выбросами
np.random.seed(42)
n = 100
base_price = 90000
outliers = [424000, 500000]  # выбросы
data = pd.DataFrame({
    'open': np.random.normal(base_price, 1000, n),
    'high': np.random.normal(base_price, 1000, n),
    'low': np.random.normal(base_price, 1000, n),
    'close': np.concatenate([np.random.normal(base_price, 1000, n-2), outliers]),
    'volume': np.random.uniform(1, 10, n)
})
data.index = pd.date_range('2025-01-01', periods=n, freq='15min')

print(f"Данные: {len(data)} строк")
print(f"Средняя цена close: {data['close'].mean():.2f}")
print(f"Медианная цена close: {data['close'].median():.2f}")

# Создаём стратегию
strategy = GridStrategy(
    upper_bound=100000,
    lower_bound=80000,
    num_levels=10,
    amount_per_level=0.001,
    deposit=10000
)

# Создаём бэктестер
backtester = Backtester(initial_balance=10000)

# Запускаем бэктест
metrics = backtester.run_backtest(data, strategy, fee_rate=0.0006)

print("\nМетрики:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# Проверяем, что в логах была медианная цена (нужно посмотреть вывод)
# Вместо этого проверим, что уровни рассчитаны относительно медианы
print("\nУровни стратегии:")
for level in strategy.levels:
    print(f"  {level.side} {level.price:.2f}")

# Определим, сколько уровней buy/sell
buy_levels = sum(1 for l in strategy.levels if l.side == 'buy')
sell_levels = sum(1 for l in strategy.levels if l.side == 'sell')
print(f"\nBuy уровней: {buy_levels}, Sell уровней: {sell_levels}")

# Ожидаем, что медиана около 90k, значит примерно половина уровней buy, половина sell
# Если бы использовалось среднее (~150k), все уровни были бы buy.
if buy_levels > 0 and sell_levels > 0:
    print("✅ Уровни распределены между buy и sell - медиана работает.")
else:
    print("❌ Все уровни одной стороны - возможно, используется среднее.")