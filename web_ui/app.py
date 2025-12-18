"""
Streamlit-интерфейс для бэктестинга Grid Trading стратегии.

Предоставляет веб-интерфейс для запуска симуляции, отображения сделок и метрик.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import os
import glob
import shutil

from config.settings import config
from services.data_loader import DataLoader
from models.grid_strategy import GridStrategy
from services.backtester import Backtester
from src.backtester_v2 import AdaptiveBacktester

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация сессионного состояния
if "trades_df" not in st.session_state:
    st.session_state.trades_df = pd.DataFrame()
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "equity_df" not in st.session_state:
    st.session_state.equity_df = pd.DataFrame()
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "data_stats" not in st.session_state:
    st.session_state.data_stats = {}
if "data_cache" not in st.session_state:
    st.session_state.data_cache = {}


def load_historical_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """
    Загружает исторические данные за указанное количество дней.
    Использует кэширование в session_state для избежания повторной загрузки.

    Args:
        symbol: Торговая пара (например, 'BTC/USDT').
        timeframe: Таймфрейм ('15m', '1h' и т.д.).
        days: Количество дней истории.

    Returns:
        DataFrame с данными.
    """
    cache_key = f"{symbol}_{timeframe}_{days}"
    # FIX (REVIEW.md): Кэширование загруженных данных в session_state для избежания повторной загрузки
    if cache_key in st.session_state.data_cache:
        logger.info(f"Используются кэшированные данные для {cache_key}")
        cached_df = st.session_state.data_cache[cache_key]
        # Обновляем статистику (может быть уже сохранена)
        if st.session_state.data_stats:
            pass
        st.session_state.data_loaded = True
        return cached_df

    try:
        loader = DataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = loader.load_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
        )
        # Предобработка
        df = loader.preprocess_data(df, fill_missing=True)
        
        # Фильтрация аномальных цен для статистики (чтобы не искажать min/max)
        df_filtered = df[(df['close'] >= 20000) & (df['close'] <= 200000)].copy()
        if len(df_filtered) < len(df):
            logger.warning(
                f"Отфильтровано {len(df) - len(df_filtered)} аномалий для статистики "
                f"(цены вне диапазона 20k-200k)"
            )
        
        # Используем отфильтрованные данные для статистики (если они не пустые)
        stats_df = df_filtered if not df_filtered.empty else df
        st.session_state.data_stats = loader.get_data_statistics(stats_df)
        
        st.session_state.data_loaded = True
        # Сохраняем в кэш
        st.session_state.data_cache[cache_key] = df
        logger.info(f"Загружено {len(df)} строк данных и сохранено в кэш.")
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        logger.exception("Ошибка загрузки данных")
        return pd.DataFrame()


def run_backtest(
    data: pd.DataFrame,
    upper_bound: float,
    lower_bound: float,
    grid_levels: int,
    order_size: float,
    initial_balance: float,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Запускает бэктест и возвращает результаты (v1.0 ручной режим).

    Args:
        data: Исторические данные.
        upper_bound: Верхняя граница сетки.
        lower_bound: Нижняя граница сетки.
        grid_levels: Количество уровней сетки.
        order_size: Размер ордера на уровень.
        initial_balance: Начальный баланс.

    Returns:
        Кортеж (trades_df, metrics, equity_df).
    """
    try:
        # Создание стратегии
        strategy = GridStrategy(
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            num_levels=grid_levels,
            amount_per_level=order_size,
            deposit=initial_balance,
        )

        # Создание бэктестера
        backtester = Backtester(initial_balance=initial_balance)

        # Запуск симуляции
        metrics = backtester.run_backtest(data, strategy)

        # Получение результатов
        trades_df = backtester.get_trades_dataframe()
        equity_df = backtester.get_equity_dataframe()

        logger.info(f"Бэктест v1.0 завершён. Сделок: {len(trades_df)}")
        return trades_df, metrics, equity_df
    except Exception as e:
        st.error(f"Ошибка выполнения бэктеста: {e}")
        logger.exception("Ошибка бэктеста")
        return pd.DataFrame(), {}, pd.DataFrame()


def run_backtest_v2(
    data: pd.DataFrame,
    use_auto_config: bool,
    initial_balance: float,
    upper_bound: float = None,
    lower_bound: float = None,
    grid_levels: int = None,
    order_size: float = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Запускает бэктест v2.0 с поддержкой автонастройки.

    Args:
        data: Исторические данные.
        use_auto_config: Использовать автонастройку.
        initial_balance: Начальный баланс.
        upper_bound: Верхняя граница (используется только при ручном режиме).
        lower_bound: Нижняя граница (используется только при ручном режиме).
        grid_levels: Количество уровней (используется только при ручном режиме).
        order_size: Размер ордера (используется только при ручном режиме).

    Returns:
        Кортеж (trades_df, metrics, equity_df).
    """
    try:
        if use_auto_config:
            # Используем AdaptiveBacktester с автонастройкой
            logger.info("Запуск бэктеста v2.0 с автонастройкой")
            backtester = AdaptiveBacktester(
                initial_balance=initial_balance,
                use_auto_config=True
            )
            
            # Для автонастройки нужны дневные данные, но у нас только торговые данные
            # Используем те же данные для упрощения
            daily_data = data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna() if len(data) > 100 else data
            
            # Запуск с автонастройкой
            metrics = backtester.run_with_auto_config(
                daily_data=daily_data,
                intraday_data=data,
                fee_rate=0.0006
            )
        else:
            # Ручной режим v1.0
            logger.info("Запуск бэктеста v2.0 в ручном режиме")
            backtester = AdaptiveBacktester(
                initial_balance=initial_balance,
                use_auto_config=False
            )
            
            # Создаем стратегию с ручными параметрами
            from src.grid_strategy_v2 import AdaptiveGridStrategy
            strategy = AdaptiveGridStrategy(
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                num_levels=grid_levels,
                amount_per_level=order_size,
                deposit=initial_balance
            )
            
            # Запуск стандартного бэктеста
            metrics = backtester.run_backtest(data, strategy)

        # Получение результатов
        trades_df = backtester.get_trades_dataframe()
        equity_df = backtester.get_equity_dataframe()

        logger.info(f"Бэктест v2.0 завершён. Сделок: {len(trades_df)}")
        return trades_df, metrics, equity_df
    except Exception as e:
        st.error(f"Ошибка выполнения бэктеста v2.0: {e}")
        logger.exception("Ошибка бэктеста v2.0")
        return pd.DataFrame(), {}, pd.DataFrame()


def display_results(trades_df: pd.DataFrame, metrics: dict, equity_df: pd.DataFrame):
    """
    Отображает результаты бэктеста: таблицы, метрики и графики.

    Args:
        trades_df: DataFrame со сделками.
        metrics: Словарь с метриками.
        equity_df: DataFrame с кривой капитала.
    """
    st.header("📊 Результаты бэктеста")

    # Метрики
    st.subheader("Ключевые метрики")
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Общая прибыль", f"{metrics.get('total_pnl', 0):.2f} USDT")
        with col2:
            st.metric("Прибыль %", f"{metrics.get('total_pnl_pct', 0):.2f}%")
        with col3:
            st.metric("Количество сделок", metrics.get("total_trades", 0))
        with col4:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0) * 100:.1f}%")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Макс. просадка", f"{metrics.get('max_drawdown', 0):.2f}%")
        with col6:
            st.metric("Коэф. Шарпа", f"{metrics.get('sharpe_ratio', 0):.2f}")
        with col7:
            st.metric("Покупки", metrics.get("buy_trades", 0))
        with col8:
            st.metric("Продажи", metrics.get("sell_trades", 0))

        st.metric("Начальный баланс", f"{metrics.get('initial_balance', 0):.2f} USDT")
        st.metric("Финальный капитал", f"{metrics.get('final_equity', 0):.2f} USDT")
        
        # Отображение режима рынка и автонастроенных параметров (v2.0)
        if 'configuration' in metrics and metrics['configuration']:
            config_data = metrics['configuration']
            mode = config_data.get('mode', 'UNKNOWN')
            mode_emoji = {"UPTREND": "📈", "DOWNTREND": "📉", "RANGE": "↔️"}
            emoji = mode_emoji.get(mode, "❓")
            
            st.subheader("🤖 Автонастроенные параметры")
            st.metric("Режим рынка", f"{emoji} {mode}")
            
            # Таблица параметров
            params_df = pd.DataFrame({
                'Параметр': ['Режим', 'Границы', 'Уровни', 'Шаг', 'Размер', 'Волатильность'],
                'Значение': [
                    mode,
                    f"{config_data.get('lower_bound', 0):.0f} - {config_data.get('upper_bound', 0):.0f}",
                    config_data.get('num_levels', 0),
                    f"{config_data.get('grid_step', 0):.2f}",
                    f"{config_data.get('amount_per_level', 0):.4f} BTC",
                    f"{config_data.get('volatility', 0):.2%}" if config_data.get('volatility') else "N/A"
                ]
            })
            st.table(params_df)
            
            # Дополнительная информация
            if config_data.get('auto_configured'):
                st.info("✅ Параметры настроены автоматически на основе анализа рынка")
            else:
                st.warning("⚠️ Используются ручные параметры")
    else:
        st.warning("Метрики недоступны.")

    # График equity curve
    st.subheader("📈 Кривая капитала")
    if not equity_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_df["timestamp"],
                y=equity_df["equity"],
                mode="lines",
                name="Капитал",
                line=dict(color="blue"),
            )
        )
        fig.update_layout(
            title="Динамика капитала",
            xaxis_title="Время",
            yaxis_title="Капитал (USDT)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для графика капитала.")

    # Таблица сделок
    st.subheader("📋 Сделки")
    if not trades_df.empty:
        # Преобразование столбцов для удобства
        display_cols = [
            "timestamp",
            "side",
            "price",
            "amount",
            "cost",
            "fee",
            "balance_after",
            "position_after",
        ]
        available_cols = [c for c in display_cols if c in trades_df.columns]
        st.dataframe(trades_df[available_cols], use_container_width=True)
    else:
        st.info("Сделок не было.")

    # Экспорт результатов
    st.subheader("📥 Экспорт результатов")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        if not trades_df.empty:
            csv_trades = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать сделки (CSV)",
                data=csv_trades,
                file_name="trades.csv",
                mime="text/csv",
            )
        else:
            st.info("Нет сделок для экспорта")
    with export_col2:
        if not equity_df.empty:
            csv_equity = equity_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать кривую капитала (CSV)",
                data=csv_equity,
                file_name="equity_curve.csv",
                mime="text/csv",
            )
        else:
            st.info("Нет кривой капитала для экспорта")

    # Дополнительная статистика
    st.subheader("📈 Дополнительная статистика")
    if not equity_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Распределение сделок по сторонам:**")
            if "side" in trades_df.columns:
                side_counts = trades_df["side"].value_counts()
                st.bar_chart(side_counts)
        with col2:
            st.write("**Объём сделок:**")
            if "amount" in trades_df.columns:
                st.line_chart(trades_df.set_index("timestamp")["amount"])
    else:
        st.info("Нет данных для дополнительной статистики.")


def main():
    """Основная функция Streamlit-приложения."""
    st.set_page_config(
        page_title="Grid Trading Backtester",
        page_icon="📊",
        layout="wide",
    )

    st.title("🤖 Grid Trading Backtester")
    st.markdown(
        """
        Этот инструмент позволяет провести бэктестирование стратегии Grid Trading
        на исторических данных с Bybit.
        """
    )

    # Боковая панель с настройками
    st.sidebar.header("⚙️ Параметры")

    # Выбор символа и таймфрейма
    symbol = st.sidebar.text_input("Торговая пара", value=config.symbol)
    timeframe = st.sidebar.selectbox(
        "Таймфрейм",
        options=["15m", "30m", "1h", "4h", "1d"],
        index=0,
    )
    days = st.sidebar.slider(
        "Дней истории",
        min_value=1,
        max_value=365,
        value=30,
        help="Количество дней исторических данных для загрузки.",
    )

    # Grid Bot v2.0
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Grid Bot v2.0")

    use_auto_config = st.sidebar.checkbox(
        "Использовать автонастройку",
        value=True,
        help="Автоматически рассчитать все параметры"
    )

    if use_auto_config:
        st.sidebar.info("✅ Автонастройка включена")
    else:
        st.sidebar.warning("⚠️ Ручной режим v1.0")

    # Параметры сетки (показываются только при выключенной автонастройке)
    st.sidebar.subheader("Параметры сетки")
    
    if not use_auto_config:
        lower_bound = st.sidebar.number_input(
            "Нижняя граница (USDT)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="Нижняя цена сетки.",
        )
        upper_bound = st.sidebar.number_input(
            "Верхняя граница (USDT)",
            min_value=0.0,
            value=60000.0,
            step=1000.0,
            help="Верхняя цена сетки.",
        )
        grid_levels = st.sidebar.slider(
            "Количество уровней",
            min_value=5,
            max_value=100,
            value=config.grid_levels,
        )
        order_size = st.sidebar.number_input(
            "Размер ордера (BTC)",
            min_value=0.001,
            value=config.order_size,
            step=0.001,
            format="%.3f",
            help="Объём на каждый уровень в BTC.",
        )
    else:
        # При автонастройке используем значения по умолчанию (будут переопределены)
        lower_bound = 50000.0
        upper_bound = 60000.0
        grid_levels = config.grid_levels
        order_size = config.order_size

    # Начальный баланс
    initial_balance = st.sidebar.number_input(
        "Начальный баланс (USDT)",
        min_value=100.0,
        value=config.initial_balance,
        step=100.0,
    )

    # Кнопка загрузки данных
    st.sidebar.subheader("Данные")
    if st.sidebar.button("📥 Загрузить исторические данные", type="secondary"):
        with st.spinner("Загрузка данных..."):
            data = load_historical_data(symbol, timeframe, days)
            if not data.empty:
                st.success(f"Загружено {len(data)} строк.")
                st.session_state.data = data
            else:
                st.error("Не удалось загрузить данные.")

    # FIX (REVIEW.md): Всегда показываем раздел "Статистика данных", с сообщением если данные не загружены
    st.sidebar.subheader("📊 Статистика данных")
    if st.session_state.data_loaded and st.session_state.data_stats:
        stats = st.session_state.data_stats
        st.sidebar.write(f"**Период:** {stats.get('start_date')} - {stats.get('end_date')}")
        st.sidebar.write(f"**Строк:** {stats.get('total_rows')}")
        st.sidebar.write(f"**Диапазон цены:** {stats.get('price_range', {}).get('min'):.2f} - {stats.get('price_range', {}).get('max'):.2f}")
        
        # Добавляем медиану и среднюю цену
        median_price = stats.get('median_price')
        mean_price = stats.get('mean_price')
        if median_price is not None and mean_price is not None:
            st.sidebar.write(f"**Медианная цена:** {median_price:.2f}")
            st.sidebar.write(f"**Средняя цена:** {mean_price:.2f}")
            
            # Опционально - показать разницу если она большая
            if median_price != 0 and abs(mean_price - median_price) / median_price > 0.1:
                st.sidebar.warning("⚠️ Средняя и медиана сильно отличаются - возможны выбросы в данных")
    else:
        st.sidebar.info("Данные не загружены.")

    # Управление кэшем
    st.sidebar.subheader("🗃️ Управление кэшем")
    confirm_clear = st.sidebar.checkbox("Подтвердить очистку кэша")
    if st.sidebar.button("🗑️ Очистить кэш данных", type="secondary", disabled=not confirm_clear):
        cache_dir = "./cache"
        if os.path.exists(cache_dir):
            pkl_files = glob.glob(os.path.join(cache_dir, "*.pkl"))
            deleted_count = 0
            for f in pkl_files:
                try:
                    os.remove(f)
                    deleted_count += 1
                except Exception as e:
                    st.sidebar.error(f"Ошибка удаления {f}: {e}")
            if deleted_count > 0:
                st.sidebar.success(f"Удалено {deleted_count} файлов кэша.")
            else:
                st.sidebar.info("Нет файлов .pkl для удаления.")
        else:
            st.sidebar.info("Директория кэша не существует.")
    if not confirm_clear:
        st.sidebar.warning("Для очистки кэша отметьте чекбокс подтверждения.")

    # Кнопка запуска бэктеста
    st.sidebar.subheader("Запуск")
    run_button = st.sidebar.button("🚀 Запустить бэктест", type="primary")

    # Основная область
    if run_button:
        if not st.session_state.data_loaded:
            st.warning("Сначала загрузите исторические данные.")
        else:
            with st.spinner("Выполняется бэктестирование..."):
                data = st.session_state.get("data")
                if data is None or data.empty:
                    st.error("Нет данных для бэктеста.")
                    return

                # Выбор версии бэктеста в зависимости от режима
                if use_auto_config:
                    # v2.0 с автонастройкой
                    trades_df, metrics, equity_df = run_backtest_v2(
                        data=data,
                        use_auto_config=True,
                        initial_balance=initial_balance,
                        # Параметры для ручного режима не нужны
                    )
                else:
                    # v1.0 ручной режим
                    trades_df, metrics, equity_df = run_backtest_v2(
                        data=data,
                        use_auto_config=False,
                        initial_balance=initial_balance,
                        upper_bound=upper_bound,
                        lower_bound=lower_bound,
                        grid_levels=grid_levels,
                        order_size=order_size,
                    )

                # Сохранение в session_state
                st.session_state.trades_df = trades_df
                st.session_state.metrics = metrics
                st.session_state.equity_df = equity_df

                st.success("Бэктест завершён!")

    # Отображение результатов (если есть)
    if not st.session_state.trades_df.empty or st.session_state.metrics:
        display_results(
            st.session_state.trades_df,
            st.session_state.metrics,
            st.session_state.equity_df,
        )
    else:
        st.info("👈 Задайте параметры и нажмите 'Запустить бэктест' для начала.")

    # Футер
    st.sidebar.markdown("---")
    st.sidebar.caption("Grid Trading Bot • Bybit • 2025")


if __name__ == "__main__":
    main()