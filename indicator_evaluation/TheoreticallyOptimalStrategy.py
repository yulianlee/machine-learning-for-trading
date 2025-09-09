import pandas as pd
import datetime as dt
from marketsimcode import compute_portvals
from marketsimcode import get_adjusted_closing_prices
import matplotlib.pyplot as plt

def author():
    return "yhern3" 

def study_group():
    return "yhern3"

def save_plot(portvals_optimal, portvals_benchmark, filename='images/theoretically_optimal_comparison.png'):
    portvals_optimal_normalized = portvals_optimal / portvals_optimal.iloc[0]
    portvals_benchmark_normalized = portvals_benchmark / portvals_benchmark.iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(portvals_optimal_normalized, 'red', label='Optimal Strategy')
    plt.plot(portvals_benchmark_normalized, 'purple', label='Benchmark')
    plt.title('Comparison of Theoretically Optimal Strategy vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(filename)


def testPolicy(symbol="JPM", 
               sd=dt.datetime(2010, 1, 1),
               ed=dt.datetime(2011, 12, 31), 
               sv=100000):
    prices = get_adjusted_closing_prices([symbol], sd, ed)
    prices = prices[symbol]
    print(prices.head())
    
    trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
    
    current_position = 0
    
    for i in range(len(prices.index) - 1):
        current_date = prices.index[i]
        next_date = prices.index[i + 1]
        
        current_price = prices.loc[current_date]
        next_price = prices.loc[next_date]

        if next_price > current_price:
            target_position = 1000 # long is price increases 
        elif next_price < current_price:
            target_position = -1000 # short if price decreases
        else:
            target_position = current_position # if prices dont change, hold
        
        shares_to_trade = target_position - current_position
        
        if shares_to_trade != 0:
            trades.loc[current_date, symbol] = shares_to_trade
        
        current_position = target_position

    # close position on last day
    if current_position != 0:
        trades.loc[prices.index[-1], symbol] = -current_position
    
    return trades

def calculate_stats(portvals):
    daily_returns = portvals.pct_change().dropna()
    cumulative_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    std_daily_return = daily_returns.std()
    mean_daily_return = daily_returns.mean()
    return cumulative_return, std_daily_return, mean_daily_return


def run_optimal(symbol = 'JPM', 
                start_date = dt.datetime(2008, 1, 1), 
                end_date = dt.datetime(2009, 12, 31),
                start_val = 100000
                ):
    # optimal strategy
    trades_df = testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_val)
    # to convert our trades into orders
    orders_list = []
    for date, row in trades_df.iterrows():
        shares = row[symbol]
        if shares > 0:
            order_type = 'BUY'
        elif shares < 0: 
            order_type = 'SELL'
        elif shares == 0:
            order_type = 'HOLD'

        orders_list.append({
            'Symbol': symbol,
            'Order': order_type,
            'Shares': abs(int(shares)),
            'Date': date
        })
    
    orders = pd.DataFrame(orders_list)

    if not orders.empty:
        orders.set_index('Date', inplace=True)
    
    benchmark_orders = pd.DataFrame(
        {'Symbol': [symbol] * len(orders), 
         'Order': ['BUY'] + ['HOLD'] * (len(orders) - 1), 
         'Shares': [1000] * len(orders) , 
         'Date': orders.index}
    )

    benchmark_orders.set_index('Date', inplace=True)
    
    portvals_optimal = compute_portvals(
        orders,
        start_val=start_val,
        commission=0.0,
        impact=0.0
    )
    
    portvals_benchmark = compute_portvals(
        benchmark_orders,
        start_val=start_val,
        commission=0.0,
        impact=0.0
    )
    
    cr_optimal, std_optimal, mean_optimal = calculate_stats(portvals_optimal)
    cr_benchmark, std_benchmark, mean_benchmark = calculate_stats(portvals_benchmark)
    
    results = f"""
    Theoretically Optimal Strategy:
    ---------------------------------
    Cumulative Return: {cr_optimal:.6f}
    Standard Deviation of Daily Returns: {std_optimal:.6f}
    Mean of Daily Returns: {mean_optimal:.6f}

    Benchmark (Optimal):
    --------------------
    Cumulative Return: {cr_benchmark:.6f}
    Standard Deviation of Daily Returns: {std_benchmark:.6f}
    Mean of Daily Returns: {mean_benchmark:.6f}
    """
    with open("p6_results.txt", "a") as f:
        f.write(results)

    save_plot(portvals_optimal, portvals_benchmark)


if __name__ == "__main__":
    run_optimal()
    

