import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from StrategyLearner import StrategyLearner
import marketsimcode as msc
import util

def author():
    return 'yhern3'

def study_group():
    return 'yhern3'

def run_experiment2(sv=100000):
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    commission = 0.00

    impact_values = [0.0, 0.01, 0.02, 0.03, 0.04]
    results = {'impact': [], 'num_trades': [], 'cum_return': []}

    for impact in impact_values:

        learner = StrategyLearner(commission=commission, impact=impact, verbose=False)
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)

        trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        portvals = msc.compute_portvals(trades, start_val=sv, commission=commission, impact=impact)

        if isinstance(portvals, pd.DataFrame): portvals = portvals.iloc[:, 0]

        num_trades = len(trades.loc[trades[symbol] != 0])
        cum_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1 if len(portvals)>0 else 0

        results['impact'].append(impact)
        results['num_trades'].append(num_trades)
        results['cum_return'].append(cum_return)

    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 5))
    plt.bar(results_df['impact'], results_df['num_trades'], width=0.001)
    plt.title(f"Experiment 2: Impact vs. Number of Trades ({symbol} In-Sample)")
    plt.xlabel("Market Impact Value")
    plt.ylabel("Number of Trades Executed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/experiment2_impact_vs_trades.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(results_df['impact'], results_df['cum_return'], width=0.001)
    plt.title(f"Experiment 2: Impact vs. Cumulative Return ({symbol} In-Sample)")
    plt.xlabel("Market Impact Value")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/experiment2_impact_vs_return.png")
    plt.close()

    return results_df

if __name__ == "__main__":
    run_experiment2()