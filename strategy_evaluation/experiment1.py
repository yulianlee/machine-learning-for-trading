import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import marketsimcode as msc
import util

def author():
    return 'yhern3'

def study_group():
    return 'yhern3'

def create_benchmark_trades(symbol, index):
    trades_df = pd.DataFrame(0, index=index, columns=[symbol])
    trades_df.iloc[0] = 1000 
    return trades_df

def run_experiment1(symbol = 'JPM', sv=100000, commission=9.95, impact=0.005):
    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)

    manual_strategy = ManualStrategy(commission=commission, impact=impact)
    strategy_learner = StrategyLearner(commission=commission, impact=impact, verbose=False)

    strategy_learner.find_best_parameters(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    strategy_learner.add_evidence(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    manual_trades_is = manual_strategy.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    learner_trades_is = strategy_learner.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    benchmark_trades_is = create_benchmark_trades(symbol=symbol, index = manual_trades_is.index)
    manual_portvals_is = msc.compute_portvals(manual_trades_is, start_val=sv, commission=commission, impact=impact)
    learner_portvals_is = msc.compute_portvals(learner_trades_is, start_val=sv, commission=commission, impact=impact)
    benchmark_portvals_is = msc.compute_portvals(benchmark_trades_is, start_val=sv, commission=commission, impact=impact) 

    manual_normed_is = manual_portvals_is / manual_portvals_is.iloc[0]
    learner_normed_is = learner_portvals_is / learner_portvals_is.iloc[0]
    benchmark_normed_is = benchmark_portvals_is / benchmark_portvals_is.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(manual_normed_is.index, manual_normed_is, label="Manual Strategy", color="red")
    plt.plot(learner_normed_is.index, learner_normed_is, label="Strategy Learner", color="blue")
    plt.plot(benchmark_normed_is.index, benchmark_normed_is, label="Benchmark", color="purple")
    plt.title(f"Experiment 1: Manual vs Learner vs Benchmark (In-Sample {symbol})")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/experiment1_in_sample.png")
    plt.close()

    manual_trades_oos = manual_strategy.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)
    learner_trades_oos = strategy_learner.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)
    benchmark_trades_oos = create_benchmark_trades(symbol=symbol, index = manual_trades_oos.index)

    manual_portvals_oos = msc.compute_portvals(manual_trades_oos, start_val=sv, commission=commission, impact=impact)
    learner_portvals_oos = msc.compute_portvals(learner_trades_oos, start_val=sv, commission=commission, impact=impact)
    benchmark_portvals_oos = msc.compute_portvals(benchmark_trades_oos, start_val=sv, commission=commission, impact=impact)

    manual_normed_oos = manual_portvals_oos / manual_portvals_oos.iloc[0]
    learner_normed_oos = learner_portvals_oos / learner_portvals_oos.iloc[0]
    benchmark_normed_oos = benchmark_portvals_oos / benchmark_portvals_oos.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(manual_normed_oos.index, manual_normed_oos, label="Manual Strategy", color="red")
    plt.plot(learner_normed_oos.index, learner_normed_oos, label="Strategy Learner", color="blue")
    plt.plot(benchmark_normed_oos.index, benchmark_normed_oos, label="Benchmark", color="purple")
    plt.title(f"Experiment 1: Manual vs Learner vs Benchmark (Out-of-Sample {symbol})")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/experiment1_out_of_sample.png")
    plt.close()

if __name__ == "__main__":
    run_experiment1()