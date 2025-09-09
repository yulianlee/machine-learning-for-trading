import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from marketsimcode import compute_portvals
from marketsimcode import get_adjusted_closing_prices

def author():
    return "yhern3"

def study_group():
    return "yhern3"

def compute_bollinger_bands(prices, window=20):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return upper_band, lower_band

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).fillna(0).rolling(window).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))

def compute_cci(prices, window=20):
    sma = prices.rolling(window).mean()
    md = prices.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    md_safe = md.replace(0, np.finfo(float).eps)
    cci = (prices - sma) / (0.015 * md_safe)

    return cci

def compute_momentum(prices, window=10):
    return (prices / prices.shift(window).replace(0, np.finfo(float).eps)) - 1

def compute_percentage_price_oscillator(prices, short_window=12, long_window=26):
    ema_short = prices.ewm(span=short_window, adjust=False).mean()
    ema_long = prices.ewm(span=long_window, adjust=False).mean()
    ppo = (ema_short - ema_long) / ema_long * 100
    return ppo

def plot_individual_indicators(prices, indicators, indicator_names):
    bollinger_indicators = {}
    other_indicators = []

    for indicator, name in zip(indicators, indicator_names):
        if name in ["Bollinger Upper", "Bollinger Lower"]:
            bollinger_indicators[name] = indicator
        else:
            other_indicators.append((indicator, name))

    if "Bollinger Upper" in bollinger_indicators and "Bollinger Lower" in bollinger_indicators:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prices, label="Price", color='blue')
        ax.plot(bollinger_indicators["Bollinger Upper"], label="Bollinger Upper", color='red')
        ax.plot(bollinger_indicators["Bollinger Lower"], label="Bollinger Lower", color='green')
        ax.set_title("Bollinger Bands")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid()

        plt.savefig("images/bollinger_bands.png")
        plt.close()

    for indicator, name in other_indicators:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prices, label="Price", color='blue')
        ax.plot(indicator, label=name, color='orange')
        ax.set_title(name)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid()

        plt.savefig(f"images/{name.replace(' ', '_').lower()}.png")
        plt.close()


def generate_signals(prices, upper_band, lower_band,  rsi, cci, momentum, ppo):
    common_index = prices.index
    for indicator in [upper_band, lower_band, rsi, cci, momentum, ppo]:
        common_index = common_index.intersection(indicator.index)
    
    df = pd.DataFrame({
        'Price': prices.loc[common_index],
        'upper_band': upper_band.loc[common_index],
        'lower_band': lower_band.loc[common_index],
        'rsi': rsi.loc[common_index],
        'cci': cci.loc[common_index],
        'momentum': momentum.loc[common_index],
        'ppo': ppo.loc[common_index]
    })
    
    buy_condition = (
        (df['lower_band'] > df['Price']) | 
        (df['rsi'] < 30) | 
        (df['cci'] < -100) | 
        ((df['momentum'] > 0) & (df['momentum'].diff() > 0)) |
        ((df['ppo'] < 20) & (df['ppo'].diff() > 0))
    )
    
    sell_condition = (
        (df['upper_band'] < df['Price']) | 
        (df['rsi'] > 70) | 
        (df['cci'] > 100) | 
        ((df['momentum'] < 0) & (df['momentum'].diff() < 0)) |
        ((df['ppo'] > 80) & (df['ppo'].diff() < 0))
    )

    # default signal value 0 (Hold)
    signals = pd.DataFrame(index=df.index)
    signals["Signal"] = 0
    signals.loc[buy_condition, "Signal"] = 1  # Buy
    signals.loc[sell_condition, "Signal"] = -1  # Sell
    
    # If buy and sell conditions are both true, prioritize sell
    signals.loc[buy_condition & sell_condition, "Signal"] = -1
    
    return signals

def signals_to_trades(signals, symbol, max_shares=1000):
    trades = pd.DataFrame(0, index=signals.index, columns=[symbol])
    current_position = 0
    
    for date in signals.index:
        signal = signals.loc[date, "Signal"]
        
        if signal == 1 and current_position <= 0:  # Buy
            shares_to_trade = max_shares - current_position
            trades.loc[date, symbol] = shares_to_trade
            current_position = max_shares
        elif signal == -1 and current_position >= 0:  # Sell
            shares_to_trade = -max_shares - current_position
            trades.loc[date, symbol] = shares_to_trade
            current_position = -max_shares
    
    # close position on last day
    if current_position != 0:
        trades.loc[signals.index[-1], symbol] = -current_position
    
    return trades

def calculate_stats(portvals):
    daily_returns = portvals.pct_change().dropna()
    cumulative_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    std_daily_return = daily_returns.std()
    mean_daily_return = daily_returns.mean()
    return cumulative_return, std_daily_return, mean_daily_return

def trades_to_orders(trades, symbol):
    orders_list = []
    for date, row in trades.iterrows():
        shares = row[symbol]
        if shares > 0:
            order_type = 'BUY'
        elif shares < 0:
            order_type = 'SELL'
        else:
            # continue  # Skip zero trades
            order_type = 'HOLD'
            
        orders_list.append({
            'Symbol': symbol,
            'Order': order_type,
            'Shares': abs(int(shares)),
            'Date': date
        })
    
    if not orders_list:
        return pd.DataFrame(columns=['Symbol', 'Order', 'Shares', 'Date'])
        
    orders = pd.DataFrame(orders_list)
    orders.set_index('Date', inplace=True)
    return orders

def create_benchmark_orders(symbol, orders):
    benchmark_orders = pd.DataFrame(
        {'Symbol': [symbol] * len(orders),
         'Order': ['BUY'] + ['HOLD'] * (len(orders) - 1),
         'Shares': [1000] * len(orders),
         'Date':  orders.index}
    )
    benchmark_orders.set_index('Date', inplace=True)
    return benchmark_orders

def save_plot(portvals_strategy, portvals_benchmark, symbol, filename='images/comparison.png'):
    portvals_strategy_normalized = portvals_strategy / portvals_strategy.iloc[0]
    portvals_benchmark_normalized = portvals_benchmark / portvals_benchmark.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(portvals_strategy_normalized, 'red', label='Technical Indicators Strategy')
    plt.plot(portvals_benchmark_normalized, 'blue', label='Benchmark (Buy & Hold)')
    plt.title(f'Strategy Performance vs Benchmark for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    prices = get_adjusted_closing_prices([symbol], sd, ed)[symbol]
    upper_band, lower_band = compute_bollinger_bands(prices)
    rsi = compute_rsi(prices)
    cci = compute_cci(prices)
    momentum = compute_momentum(prices)
    ppo = compute_percentage_price_oscillator(prices)

    signals = generate_signals(prices, upper_band, lower_band, rsi, cci, momentum, ppo)

    indicators = [upper_band, lower_band, rsi, cci, momentum, ppo]
    indicator_names = ["Bollinger Upper","Bollinger Lower", "RSI", "CCI", "Momentum", "Percentage Price Oscillator"]
    plot_individual_indicators(prices, indicators, indicator_names)

    # Plot signals
    plt.figure(figsize=(14, 7))
    plt.plot(prices, label="Price", color="blue")
    
    if (signals["Signal"] == 1).any():
        plt.scatter(signals.index[signals["Signal"] == 1], 
                    prices.loc[signals.index[signals["Signal"] == 1]], 
                    label="Buy Signal", marker="^", color="green", s=100)
    
    if (signals["Signal"] == -1).any():
        plt.scatter(signals.index[signals["Signal"] == -1], 
                    prices.loc[signals.index[signals["Signal"] == -1]], 
                    label="Sell Signal", marker="v", color="red", s=100)
    
    plt.title(f"Trading Signals for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig("images/signals.png")
    plt.close()

    trades = signals_to_trades(signals, symbol)
    orders = trades_to_orders(trades, symbol)
    benchmark_orders = create_benchmark_orders(symbol, orders)

    if not orders.empty:
        portvals_strategy = compute_portvals(
            orders,
            start_val=sv,
            commission=0.0,
            impact=0.0
        )
    else:
        portvals_strategy = pd.Series(sv, index=prices.index)
    
    portvals_benchmark = compute_portvals(
        benchmark_orders,
        start_val=sv,
        commission=0.0,
        impact=0.0
    )

    cr_strategy, std_strategy, mean_strategy = calculate_stats(portvals_strategy)
    cr_benchmark, std_benchmark, mean_benchmark = calculate_stats(portvals_benchmark)
    
    results_strategy = f"""
    Technical Indicators Strategy:
    ------------------------------
    Cumulative Return: {cr_strategy:.6f}
    Standard Deviation of Daily Returns: {std_strategy:.6f}
    Mean of Daily Returns: {mean_strategy:.6f}

    Benchmark (Indicators Strategy):
    ---------------------------------
    Cumulative Return: {cr_benchmark:.6f}
    Standard Deviation of Daily Returns: {std_benchmark:.6f}
    Mean of Daily Returns: {mean_benchmark:.6f}
    """

    with open("p6_results.txt", "a") as f:
        f.write(results_strategy)
    
    save_plot(portvals_strategy, portvals_benchmark, f"images/{symbol}")

    return signals

if __name__ == "__main__":
    signals = main()