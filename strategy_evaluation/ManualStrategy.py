import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data


class ManualStrategy:
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.bb_window = 20
        self.rsi_window = 14
        self.ppo_short_window = 12
        self.ppo_long_window = 26
        self.ppo_signal_window = 9

    def _compute_bollinger_bands_pct(self, prices, window):
        sma = prices.rolling(window=window, min_periods=window).mean()
        std = prices.rolling(window=window, min_periods=window).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        bb_pct = (prices - lower_band) / (upper_band - lower_band)
        return bb_pct

    def _compute_rsi(self, prices, window):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=window, min_periods=window).mean()
        avg_loss = loss.ewm(span=window, min_periods=window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _compute_ppo(self, prices, short_window, long_window):
        short_ema = prices.ewm(span=short_window, adjust=False).mean()
        long_ema = prices.ewm(span=long_window, adjust=False).mean()
        ppo = ((short_ema - long_ema) / long_ema) * 100
        return ppo

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[symbol].ffill().bfill()

        bb_pct = self._compute_bollinger_bands_pct(prices, window=self.bb_window)
        rsi = self._compute_rsi(prices, window=self.rsi_window)
        ppo = self._compute_ppo(prices, short_window=self.ppo_short_window, long_window=self.ppo_long_window)

        # define thresholds
        BB_LOWER_THRESH = 0.15  # long when price near lower band
        BB_UPPER_THRESH = 0.85  # short when price near upper band
        RSI_LOWER_THRESH = 35   # long when oversold
        RSI_UPPER_THRESH = 65   # short when overbought
        PPO_BUY_THRESH = 0.5   # long on positive momentum
        PPO_SELL_THRESH = -0.5  # short on negative momentum

        COMBINED_LONG_ENTRY = 2
        COMBINED_SHORT_ENTRY = -2
        COMBINED_EXIT = 1

        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        current_position = 0

        lookback = max(self.bb_window, self.rsi_window, self.ppo_long_window) + 1 

        for i in range(lookback, len(prices)):
            date = prices.index[i]

            if pd.isna(bb_pct.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(ppo.iloc[i]):
                continue

            bb_signal = 0
            if bb_pct.iloc[i] < BB_LOWER_THRESH: bb_signal = 1
            elif bb_pct.iloc[i] > BB_UPPER_THRESH: bb_signal = -1

            rsi_signal = 0
            if rsi.iloc[i] < RSI_LOWER_THRESH: rsi_signal = 1
            elif rsi.iloc[i] > RSI_UPPER_THRESH: rsi_signal = -1

            ppo_signal = 0
            if ppo.iloc[i] > PPO_BUY_THRESH: ppo_signal = 1
            elif ppo.iloc[i] < PPO_SELL_THRESH: ppo_signal = -1
            combined_signal = bb_signal + rsi_signal + ppo_signal

            trade_amount = 0
            if current_position == 0:
                if combined_signal >= COMBINED_LONG_ENTRY:
                    trade_amount = 1000  # enter long
                    current_position = 1000
                elif combined_signal <= COMBINED_SHORT_ENTRY:
                    trade_amount = -1000 # enter short
                    current_position = -1000
            
            elif current_position == 1000: 
                if combined_signal < COMBINED_EXIT:
                    trade_amount = -1000 # exit long
                    current_position = 0
            elif current_position == -1000:
                if combined_signal > -COMBINED_EXIT:
                     trade_amount = 1000 # exit short
                     current_position = 0

            if trade_amount != 0:
                trades.loc[date, symbol] = trade_amount

        return trades

    def evaluate_performance(self, trades, prices, sv):
        if isinstance(prices, pd.DataFrame):
            prices = prices[prices.columns[0]]
        
        common_dates = trades.index.intersection(prices.index)
        trades = trades.loc[common_dates]
        prices = prices.loc[common_dates]
        
        holdings = trades.cumsum()
        symbol = trades.columns[0]
        
        commission_costs = pd.Series(0.0, index=trades.index)
        commission_costs[trades[symbol] != 0] = self.commission
        impact_costs = abs(trades[symbol] * prices * self.impact)
        transaction_costs = commission_costs + impact_costs
        cash = pd.Series(sv, index=trades.index)
        cash = cash - (trades[symbol] * prices).cumsum() - transaction_costs.cumsum()
        stock_value = holdings[symbol] * prices
        portfolio_value = cash + stock_value
        
        normed = portfolio_value / portfolio_value.iloc[0]
        daily_returns = normed.pct_change(fill_method=None).fillna(0)
        
        avg_daily_ret = daily_returns.mean()
        std_daily_ret = daily_returns.std()
        
        sharpe_ratio = (avg_daily_ret / std_daily_ret) * np.sqrt(252) if std_daily_ret > 0 else 0
        cum_ret = normed.iloc[-1] - 1
        
        return normed, avg_daily_ret, std_daily_ret, sharpe_ratio, cum_ret
    
    def plot_performance(self, symbol, trades, prices, sv, title, filename):
        benchmark_trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        benchmark_trades.iloc[0] = 1000  # Buy 1000 shares on first day
        
        strategy_normed, _, _, _, _ = self.evaluate_performance(trades, prices, sv)
        benchmark_normed, _, _, _, _ = self.evaluate_performance(benchmark_trades, prices, sv)
        
        plt.figure(figsize=(10, 6))
        plt.plot(strategy_normed, label='Manual Strategy', color='red')
        plt.plot(benchmark_normed, label='Benchmark', color='purple')
        
        long_entries = trades[trades[symbol] > 0].index
        short_entries = trades[trades[symbol] < 0].index
        
        for date in long_entries:
            plt.axvline(x=date, color='blue', linestyle='--', linewidth=1)
        
        for date in short_entries:
            plt.axvline(x=date, color='black', linestyle='--', linewidth=1)
        
        plt.xlabel('Date')
        plt.ylabel('Normalized Value')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        plt.savefig(filename)
        plt.close()
    
    def author(self):
        return 'yhern3'
    
    def study_group(self):
        return 'yhern3'
    
# if __name__ == "__main__":
#     ms = ManualStrategy()
#     symbol = 'JPM'
    
#     in_start_date = dt.datetime(2008, 1, 1)
#     in_end_date = dt.datetime(2009, 12, 31)
#     sv = 100000
    
#     out_start_date = dt.datetime(2010, 1, 1)
#     out_end_date = dt.datetime(2011, 12, 31)
    
#     print("\n=== PROCESSING IN-SAMPLE PERIOD ===")
#     in_trades = ms.testPolicy(symbol, in_start_date, in_end_date, sv)
#     in_trades.to_csv('test.csv')
#     in_prices = get_data([symbol], pd.date_range(in_start_date, in_end_date))[symbol]
#     in_prices.fillna(method='ffill', inplace=True)
#     in_prices.fillna(method='bfill', inplace=True)
    
#     print("\n=== PROCESSING OUT-OF-SAMPLE PERIOD ===")
#     out_trades = ms.testPolicy(symbol, out_start_date, out_end_date, sv)
#     out_prices = get_data([symbol], pd.date_range(out_start_date, out_end_date))[symbol]
#     out_prices.fillna(method='ffill', inplace=True)
#     out_prices.fillna(method='bfill', inplace=True)
    
#     # Create benchmark for in-sample - buy 1000 shares on first day and hold
#     in_benchmark_trades = pd.DataFrame(0, index=in_prices.index, columns=[symbol])
#     in_benchmark_trades.iloc[0] = 1000  # Buy 1000 shares on first day
    
#     # Create benchmark for out-of-sample
#     out_benchmark_trades = pd.DataFrame(0, index=out_prices.index, columns=[symbol])
#     out_benchmark_trades.iloc[0] = 1000
    
#     print("\n=== EVALUATING IN-SAMPLE PERFORMANCE ===")
#     print("Manual Strategy:")
#     in_strategy_normed, in_avg_daily_ret, in_std_daily_ret, in_sharpe_ratio, in_cum_ret = ms.evaluate_performance(in_trades, in_prices, sv)
    
#     print("\nBenchmark:")
#     in_benchmark_normed, in_benchmark_avg_ret, in_benchmark_std_ret, in_benchmark_sharpe, in_benchmark_cum_ret = ms.evaluate_performance(in_benchmark_trades, in_prices, sv)
    
#     print("\n=== EVALUATING OUT-OF-SAMPLE PERFORMANCE ===")
#     print("Manual Strategy:")
#     out_strategy_normed, out_avg_daily_ret, out_std_daily_ret, out_sharpe_ratio, out_cum_ret = ms.evaluate_performance(out_trades, out_prices, sv)
    
#     print("\nBenchmark:")
#     out_benchmark_normed, out_benchmark_avg_ret, out_benchmark_std_ret, out_benchmark_sharpe, out_benchmark_cum_ret = ms.evaluate_performance(out_benchmark_trades, out_prices, sv)

#     ms.plot_performance(symbol, in_trades, in_prices, sv, 
#                         f'{symbol} Manual Strategy vs. Benchmark (In-Sample)',
#                         "images/manual_strategy_in_sample.png")
    
#     ms.plot_performance(symbol, out_trades, out_prices, sv, 
#                         f'{symbol} Manual Strategy vs. Benchmark (Out-of-Sample)',
#                         "images/manual_strategy_out_sample.png")
    
#     print("\n--- Performance Summary ---")
#     print("\nIn-Sample Period ({} to {}):".format(in_start_date.strftime('%Y-%m-%d'), in_end_date.strftime('%Y-%m-%d')))
#     print("                       Benchmark    Manual Strategy")
#     print("Cumulative Return:     {:.6f}     {:.6f}".format(in_benchmark_cum_ret, in_cum_ret))
#     print("Std Dev Daily Returns: {:.6f}     {:.6f}".format(in_benchmark_std_ret, in_std_daily_ret))
#     print("Mean Daily Returns:    {:.6f}     {:.6f}".format(in_benchmark_avg_ret, in_avg_daily_ret))
#     print("Sharpe Ratio:          {:.6f}     {:.6f}".format(in_benchmark_sharpe, in_sharpe_ratio))
    
#     print("\nOut-of-Sample Period ({} to {}):".format(out_start_date.strftime('%Y-%m-%d'), out_end_date.strftime('%Y-%m-%d')))
#     print("                       Benchmark    Manual Strategy")
#     print("Cumulative Return:     {:.6f}     {:.6f}".format(out_benchmark_cum_ret, out_cum_ret))
#     print("Std Dev Daily Returns: {:.6f}     {:.6f}".format(out_benchmark_std_ret, out_std_daily_ret))
#     print("Mean Daily Returns:    {:.6f}     {:.6f}".format(out_benchmark_avg_ret, out_avg_daily_ret))
#     print("Sharpe Ratio:          {:.6f}     {:.6f}".format(out_benchmark_sharpe, out_sharpe_ratio))
    