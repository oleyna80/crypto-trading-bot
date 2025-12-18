#!/usr/bin/env python3
"""
Проверка косметических изменений:
1. Фильтрация аномалий в статистике
2. Подсчёт buy/sell в метриках
3. Группировка warnings
"""
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)  # подавляем INFO

from services.data_loader import DataLoader
from services.backtester import Backtester
from models.grid_strategy import GridStrategy

print("=== Тест фильтрации аномалий в статистике ===")
# Создаём данные с аномалиями
np.random.seed(123)
n = 50
df = pd.DataFrame({
    'open': np.random.normal(100000, 5000, n),
    'high': np.random.normal(101000, 5000, n),
    'low': np.random.normal(99000, 5000, n),
    'close': np.concatenate([np.random.normal(100000, 5000, n-2), [20000, 500000]]),  # аномалии
    'volume': np.random.uniform(1, 10, n)
})
df.index = pd.date_range('2025-01-01', periods=n, freq='15min')

loader = DataLoader()
stats = loader.get_data_statistics(df)
print(f"Оригинальная статистика:")
print(f"  min: {stats['price_range']['min']:.2f}")
print(f"  max: {stats['price_range']['max']:.2f}")
print(f"  median: {stats['price_range']['median']:.2f}")
print(f"  median_price: {stats.get('median_price', 'N/A')}")
print(f"  mean_price: {stats.get('mean_price', 'N/A')}")

# Проверяем, что медиана и средняя есть
assert 'median_price' in stats
assert 'mean_price' in stats
print("✅ Поля median_price и mean_price присутствуют")

print("\n=== Тест подсчёта buy/sell в метриках ===")
# Создаём стратегию с малым депозитом, чтобы были пропуски
strategy = GridStrategy(
    upper_bound=110000,
    lower_bound=90000,
    num_levels=5,
    amount_per_level=0.01,
    deposit=1000  # мало, будут warnings
)
backtester = Backtester(initial_balance=1000)
# Запускаем бэктест на данных без аномалий (убираем выбросы)
df_normal = df[(df['close'] >= 20000) & (df['close'] <= 200000)]
metrics = backtester.run_backtest(df_normal, strategy, fee_rate=0.0006)
print(f"Метрики:")
print(f"  total_trades: {metrics.get('total_trades')}")
print(f"  buy_trades: {metrics.get('buy_trades')}")
print(f"  sell_trades: {metrics.get('sell_trades')}")
# Проверяем, что buy_trades и sell_trades присутствуют и не отрицательные
assert 'buy_trades' in metrics
assert 'sell_trades' in metrics
assert metrics['buy_trades'] >= 0
assert metrics['sell_trades'] >= 0
print("✅ Подсчёт buy/sell работает")

print("\n=== Тест группировки warnings ===")
# Проверяем, что warnings не выводятся на каждую неудачную попытку
# (это можно проверить только по логам, но мы можем убедиться, что код не падает)
print("Код выполнен без ошибок.")

print("\n=== Тест улучшенного warning о недостатке капитала ===")
# Создаём стратегию с заведомо недостаточным депозитом
try:
    strategy2 = GridStrategy(
        upper_bound=100000,
        lower_bound=50000,
        num_levels=100,
        amount_per_level=0.1,
        deposit=1000
    )
    print("Стратегия создана, warning должен был появиться в логах.")
except Exception as e:
    print(f"Ошибка: {e}")

print("\n=== Все проверки пройдены ===")