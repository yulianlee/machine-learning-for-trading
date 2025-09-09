""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Yulian Lee Ying Hern		  	   		 	 	 			  		 			     			  	 
GT User ID: yhern3 		  	   		 	 	 			  		 			     			  	 
GT ID: 903870865 		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util as ut
from ManualStrategy import ManualStrategy
import marketsimcode as ms

from BagLearner import BagLearner
from RTLearner import RTLearner
from util import get_data
from ManualStrategy import ManualStrategy


class StrategyLearner(object):
    """
    A strategy learner that uses a Random Forest (BagLearner with RTLearners)
    """
    
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = None
        self.N = 12  # future horizon length for label generation
        self.YBUY = 0.0125  # default value for threshold 
        self.YSELL = -0.0125  # default value for threshold 
        self.bags = 20
        self.leaf_size = 5 # default value for leaf size 
        self.min_holding_days = 1
        

    def _create_features(self, price_series):
        ms = ManualStrategy(impact=self.impact, commission=self.commission)

        bb_percentage = ms._compute_bollinger_bands_pct(price_series, window=15)
        rsi = ms._compute_rsi(price_series, window=10)
        ppo = ms._compute_ppo(price_series, short_window=10, long_window=30)

        indicators = pd.DataFrame(index=price_series.index)
        indicators['bb'] = bb_percentage
        indicators['rsi'] = rsi
        indicators['ppo'] = ppo
        indicators = indicators.dropna()
        
        return indicators
    
    def _create_labels(self, price_series, N):
        future_returns = price_series.pct_change(N).shift(-N)
        YBUY = self.YBUY 
        YSELL = self.YSELL 

        labels = pd.Series(0, index=price_series.index)
        valid_label_indices = future_returns.dropna().index
        labels.loc[valid_label_indices[future_returns[valid_label_indices] > YBUY]] = 1
        labels.loc[valid_label_indices[future_returns[valid_label_indices] < YSELL]] = -1

        return labels
        
    def add_evidence(self, 
                     symbol="JPM", 
                     sd=dt.datetime(2008, 1, 1), 
                     ed=dt.datetime(2009, 1, 1), 
                     sv=10000,
                     ):
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[[symbol]]
        price_series = prices[symbol]
        
        price_series = price_series.fillna(method='ffill')
        price_series = price_series.fillna(method='bfill')
        
        features = self._create_features(price_series)
        labels = self._create_labels(price_series, self.N)
        
        valid_indices = features.index.intersection(labels.index)
        X = features.loc[valid_indices].values
        y = labels.loc[valid_indices].values
        
        y = y.astype(int)
        
        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": self.leaf_size},
            bags=self.bags,
            boost=False,
            verbose=self.verbose
        )
        self.learner.add_evidence(X, y)
        
        if self.verbose:
            volatility = price_series.pct_change().rolling(window=20).std().mean()
            print(f"Average volatility: {volatility:.6f}")
            print(f"YBUY threshold: {self.YBUY:.6f}")
            print(f"YSELL threshold: {self.YSELL:.6f}")


    def testPolicy(self, symbol="JPM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=100000):
        if self.learner is None:
            raise ValueError("Learner has not been trained. Call add_evidence first.")

        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates) # Use alias 'ut'
        prices = prices_all[[symbol]]
        price_series = prices[symbol]

        price_series = price_series.fillna(method='ffill')
        price_series = price_series.fillna(method='bfill')

        features = self._create_features(price_series)

        X = features.values
        predictions = self.learner.query(X)
        predictions_series = pd.Series(predictions, index=features.index)
        aligned_predictions = predictions_series.reindex(prices.index).ffill()

        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        position = 0
        last_trade_date = None
        min_holding_days = self.min_holding_days 

        long_streak = 0
        short_streak = 0

        for date in prices.index:
            if pd.isna(aligned_predictions.loc[date]):
                continue

            if last_trade_date is not None:
                current_ts = pd.Timestamp(date)
                last_trade_ts = pd.Timestamp(last_trade_date)
                days_since_trade = (current_ts - last_trade_ts).days
                if days_since_trade < min_holding_days:
                    continue

            pred = int(aligned_predictions.loc[date])

            if pred == 1:
                long_streak += 1
                short_streak = 0
            elif pred == -1:
                short_streak += 1 
                long_streak = 0
            else:
                long_streak = 0
                short_streak = 0

            if self.impact >= 0.015: 
                required_streak = 3
            elif self.impact >= 0.005: 
                required_streak = 2
            else: 
                required_streak = 1

            target_position = position

            if position <= 0 and long_streak >= required_streak:
                target_position = 1000
            elif position >= 0 and short_streak >= required_streak:
                target_position = -1000

            if position == 1000 and pred == -1:
                target_position = 0
            elif position == -1000 and pred == 1:
                target_position = 0

            trade = target_position - position

            if trade != 0:
                trades.loc[date, symbol] = trade
                position = target_position
                last_trade_date = date

        return trades
            
    def find_best_parameters(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
            best_sharpe = -np.inf
            best_params_sharpe = None
            best_return = -np.inf
            
            # tuneable params
            leaf_sizes = [5, 10, 15]
            bag_counts = [20, 40]
            N_values = [5, 10, 15]
            ybuy_values = [0.01, 0.015, 0.02]
            holding_days_options = [0, 3, 5]

            results = []

            for leaf_size in leaf_sizes:
                for bags in bag_counts:
                    for N in N_values:
                        for ybuy in ybuy_values:
                            for min_holding_days in holding_days_options:
                                ysell = -ybuy
                                
                                learner = StrategyLearner(impact=self.impact, commission=self.commission)
                                learner.leaf_size = leaf_size
                                learner.bags = bags
                                learner.N = N
                                learner.YBUY = ybuy
                                learner.YSELL = ysell
                                learner.min_holding_days = min_holding_days

                                learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
                                trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

                                if len(trades) == 0:
                                    cum_return = 0.0
                                    sharpe_ratio = 0.0
                                    std_daily = 0.0
                                    num_trades = 0
                                else:
                                    portvals = ms.compute_portvals(trades, start_val=sv, commission=self.commission, impact = self.impact)
                                    daily_returns = (portvals / portvals.shift(1) - 1).iloc[1:]
                                    cum_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
                                    std_daily = daily_returns.std()
                                    sharpe_ratio = (daily_returns.mean() / std_daily) * np.sqrt(252) if std_daily > 0 else 0
                                    num_trades = len(trades)

                                results.append({
                                    'leaf_size': leaf_size, 'bags': bags, 'N': N, 'YBUY': ybuy, 
                                    'min_holding_days': min_holding_days,
                                    'return': cum_return, 'sharpe': sharpe_ratio, 
                                    'trades': num_trades, 'std_dev': std_daily
                                })

                                # find best params based on sharpe ratio
                                if sharpe_ratio > best_sharpe:
                                    best_sharpe = sharpe_ratio
                                    best_params_sharpe = {
                                        'leaf_size': leaf_size, 'bags': bags, 'N': N, 
                                        'YBUY': ybuy, 'YSELL': ysell, 
                                        'min_holding_days': min_holding_days
                                    }

                                if cum_return > best_return:
                                    best_return = cum_return

            if best_params_sharpe:
                self.leaf_size = best_params_sharpe['leaf_size']
                self.bags = best_params_sharpe['bags']
                self.N = best_params_sharpe['N']
                self.YBUY = best_params_sharpe['YBUY']
                self.YSELL = best_params_sharpe['YSELL']
                self.min_holding_days = best_params_sharpe['min_holding_days']

            # results_df = pd.DataFrame(results)
            # print("\nParameter Search Results (sorted by Sharpe):")
            # print(results_df.sort_values('sharpe', ascending=False).head(5)) 
    
    def plot_actual_vs_predicted(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        price_series = prices_all[symbol]
        
        price_series = price_series.fillna(method='ffill').fillna(method='bfill')
        features = self._create_features(price_series)
        true_labels = self._create_labels(price_series, self.N)
        valid_indices = features.index.intersection(true_labels.index)
        X = features.loc[valid_indices].values
        predictions = self.learner.query(X)

        comparison_df = pd.DataFrame(index=valid_indices)
        comparison_df['Price'] = price_series.loc[valid_indices]
        comparison_df['True_Labels'] = true_labels.loc[valid_indices]
        comparison_df['Predictions'] = predictions

        plt.figure(figsize=(14, 10))
        
        ax1 = plt.subplot(211)
        ax1.plot(comparison_df.index, comparison_df['Price'], 'b', label='Stock Price')
        ax1.set_ylabel('Price')
        ax1.set_title(f'{symbol} Price vs. Classification Labels')
        ax1.legend(loc='upper left')
        
        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(comparison_df.index, comparison_df['True_Labels'], 'g', label='True Labels')
        ax2.plot(comparison_df.index, comparison_df['Predictions'], 'r', label='Predictions')
        ax2.set_ylim([-1.5, 1.5])
        ax2.set_ylabel('Signal (-1=Short, 0=Cash, 1=Long)')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        
        ax1.grid(True)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        plt.close()

        return comparison_df
    
    def author(self):
        return 'yhern3'
    
    def study_group(self):
        return 'yhern3'
    
# if __name__ == "__main__":
#     symbol = 'JPM'
#     commission = 9.95
#     impact = 0.005

#     in_start_date = dt.datetime(2008, 1, 1)
#     in_end_date = dt.datetime(2009, 12, 31)
#     sv = 100000

#     out_start_date = dt.datetime(2010, 1, 1)
#     out_end_date = dt.datetime(2011, 12, 31)

#     print("\n=== TRAINING CLASSIFICATION STRATEGY LEARNER (IN-SAMPLE) ===")
#     learner = StrategyLearner(verbose=True, impact=impact, commission=commission)
#     learner.find_best_parameters(symbol="JPM") # leaf_size, bags, N, YBUY, YSELL

#     learner.add_evidence(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
    
#     manual_strategy = ManualStrategy(impact=impact, commission=commission)
    
#     print("\n=== PROCESSING IN-SAMPLE PERIOD ===")
#     in_trades_sl = learner.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
#     in_trades_ms = manual_strategy.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date, sv=sv)
#     in_prices = get_data([symbol], pd.date_range(in_start_date, in_end_date))[symbol]
#     in_prices.fillna(method='ffill', inplace=True)
#     in_prices.fillna(method='bfill', inplace=True)
    
#     print("\n=== PROCESSING OUT-OF-SAMPLE PERIOD ===")
#     out_trades_sl = learner.testPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date, sv=sv)
#     out_trades_ms = manual_strategy.testPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date, sv=sv)
#     out_prices = get_data([symbol], pd.date_range(out_start_date, out_end_date))[symbol]
#     out_prices.fillna(method='ffill', inplace=True)
#     out_prices.fillna(method='bfill', inplace=True)
    
#     # Create benchmark traders - buy 1000 shares on first day and hold
#     in_benchmark_trades = pd.DataFrame(0, index=in_prices.index, columns=[symbol])
#     in_benchmark_trades.iloc[0] = 1000
    
#     out_benchmark_trades = pd.DataFrame(0, index=out_prices.index, columns=[symbol])
#     out_benchmark_trades.iloc[0] = 1000
    
#     # count no. of trades
#     in_sl_trades_count = (in_trades_sl != 0).sum().sum()
#     out_sl_trades_count = (out_trades_sl != 0).sum().sum()
#     in_ms_trades_count = (in_trades_ms != 0).sum().sum()
#     out_ms_trades_count = (out_trades_ms != 0).sum().sum()
    
#     # portfolio values
#     in_portvals_sl = ms.compute_portvals(in_trades_sl, start_val=sv, commission=commission, impact=impact)
#     in_portvals_ms = ms.compute_portvals(in_trades_ms, start_val=sv, commission=commission, impact=impact)
#     in_portvals_bm = ms.compute_portvals(in_benchmark_trades, start_val=sv, commission=commission, impact=impact)
    
#     out_portvals_sl = ms.compute_portvals(out_trades_sl, start_val=sv, commission=commission, impact=impact)
#     out_portvals_ms = ms.compute_portvals(out_trades_ms, start_val=sv, commission=commission, impact=impact)
#     out_portvals_bm = ms.compute_portvals(out_benchmark_trades, start_val=sv, commission=commission, impact=impact)
    
#     # norm portfolio values
#     in_portvals_sl_norm = in_portvals_sl / in_portvals_sl[0]
#     in_portvals_ms_norm = in_portvals_ms / in_portvals_ms[0]
#     in_portvals_bm_norm = in_portvals_bm / in_portvals_bm[0]
    
#     out_portvals_sl_norm = out_portvals_sl / out_portvals_sl[0]
#     out_portvals_ms_norm = out_portvals_ms / out_portvals_ms[0]
#     out_portvals_bm_norm = out_portvals_bm / out_portvals_bm[0]
    
#     def calculate_portfolio_stats(portvals):
#         daily_returns = portvals.pct_change().dropna()
#         cum_return = (portvals[-1] / portvals[0]) - 1
#         avg_daily_return = daily_returns.mean()
#         std_daily_return = daily_returns.std()
#         sharpe_ratio = np.sqrt(252) * avg_daily_return / std_daily_return if std_daily_return > 0 else 0
#         return cum_return, std_daily_return, avg_daily_return, sharpe_ratio
    
#     # summary stats
#     in_cr_bm, in_std_bm, in_mean_bm, in_sr_bm = calculate_portfolio_stats(in_portvals_bm)
#     in_cr_ms, in_std_ms, in_mean_ms, in_sr_ms = calculate_portfolio_stats(in_portvals_ms)
#     in_cr_sl, in_std_sl, in_mean_sl, in_sr_sl = calculate_portfolio_stats(in_portvals_sl)
    
#     out_cr_bm, out_std_bm, out_mean_bm, out_sr_bm = calculate_portfolio_stats(out_portvals_bm)
#     out_cr_ms, out_std_ms, out_mean_ms, out_sr_ms = calculate_portfolio_stats(out_portvals_ms)
#     out_cr_sl, out_std_sl, out_mean_sl, out_sr_sl = calculate_portfolio_stats(out_portvals_sl)
    
#     def plot_performance_comparison(benchmark, manual, strategy_learner, title, filename):
#         plt.figure(figsize=(12, 6))
#         plt.plot(benchmark, label='Benchmark', color='blue')
#         plt.plot(manual, label='Manual Strategy', color='green')
#         plt.plot(strategy_learner, label='Strategy Learner', color='red')
#         plt.title(title)
#         plt.xlabel('Date')
#         plt.ylabel('Normalized Portfolio Value')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(filename)
#         plt.close()
     
#     plot_performance_comparison(
#         in_portvals_bm_norm, 
#         in_portvals_ms_norm, 
#         in_portvals_sl_norm, 
#         f'{symbol} Strategy Comparison (In-Sample)',
#         "images/strategy_comparison_in_sample.png"
#     )
    
#     plot_performance_comparison(
#         out_portvals_bm_norm, 
#         out_portvals_ms_norm, 
#         out_portvals_sl_norm, 
#         f'{symbol} Strategy Comparison (Out-of-Sample)',
#         "images/strategy_comparison_out_sample.png"
#     )
#     print(f"\nNumber of trades - Strategy Learner (In-sample): {in_sl_trades_count}")
#     print(f"Number of trades - Strategy Learner (Out-of-sample): {out_sl_trades_count}")
#     print(f"Number of trades - Manual Strategy (In-sample): {in_ms_trades_count}")
#     print(f"Number of trades - Manual Strategy (Out-of-sample): {out_ms_trades_count}")

#     print("\n--- Summary Table: In-Sample ---")
#     in_summary = pd.DataFrame({
#         'Benchmark': [in_cr_bm, in_std_bm, in_mean_bm, in_sr_bm, 1],
#         'Manual Strategy': [in_cr_ms, in_std_ms, in_mean_ms, in_sr_ms, in_ms_trades_count],
#         'Strategy Learner': [in_cr_sl, in_std_sl, in_mean_sl, in_sr_sl, in_sl_trades_count]
#     }, index=['Cumulative Return', 'Std Dev Daily Returns', 'Mean Daily Returns', 'Sharpe Ratio', 'Number of Trades'])
#     print(in_summary)
    
#     print("\n--- Summary Table: Out-of-Sample ---")
#     out_summary = pd.DataFrame({
#         'Benchmark': [out_cr_bm, out_std_bm, out_mean_bm, out_sr_bm, 1],
#         'Manual Strategy': [out_cr_ms, out_std_ms, out_mean_ms, out_sr_ms, out_ms_trades_count],
#         'Strategy Learner': [out_cr_sl, out_std_sl, out_mean_sl, out_sr_sl, out_sl_trades_count]
#     }, index=['Cumulative Return', 'Std Dev Daily Returns', 'Mean Daily Returns', 'Sharpe Ratio', 'Number of Trades'])
#     print(out_summary)