import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import marketsimcode as msc
import util
from experiment1 import run_experiment1
from experiment2 import run_experiment2

def author():
    return 'yhern3'

def study_group():
    return 'yhern3'

def plot_comparison(df_normed, title, filename):
    plt.figure(figsize=(12, 6))
    for col in df_normed.columns:
        plt.plot(df_normed.index, df_normed[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_benchmark_trades(symbol, index):
    trades_df = pd.DataFrame(0, index=index, columns=[symbol])
    trades_df.iloc[0] = 1000 
    return trades_df


def generate_manual_report_data(sv=100000, commission=9.95, impact=0.005):
    symbol = "JPM"
    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)

    ms = ManualStrategy(commission=commission, impact=impact)
    ms_trades_is = ms.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    bm_trades_is = create_benchmark_trades(symbol=symbol, index = ms_trades_is.index)
    ms_portvals_is = msc.compute_portvals(ms_trades_is, sv, commission, impact)
    bm_portvals_is = msc.compute_portvals(bm_trades_is, sv, commission, impact)

    df_normed_is = pd.DataFrame({
        "Manual Strategy": ms_portvals_is / ms_portvals_is.iloc[0],
        "Benchmark": bm_portvals_is / bm_portvals_is.iloc[0]
    })
    plot_comparison(df_normed_is, f"Manual Strategy vs Benchmark ({symbol} In-Sample)", "images/manual_vs_benchmark_in_sample.png")

    ms_trades_oos = ms.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)
    bm_trades_oos = create_benchmark_trades(symbol=symbol, index = ms_trades_oos.index)
    ms_portvals_oos = msc.compute_portvals(ms_trades_oos, sv, commission, impact)
    bm_portvals_oos = msc.compute_portvals(bm_trades_oos, sv, commission, impact)

    df_normed_oos = pd.DataFrame({
        "Manual Strategy": ms_portvals_oos / ms_portvals_oos.iloc[0],
        "Benchmark": bm_portvals_oos / bm_portvals_oos.iloc[0]
    })
    plot_comparison(df_normed_oos, f"Manual Strategy vs Benchmark ({symbol} Out-of-Sample)", "images/manual_vs_benchmark_out_of_sample.png")

def generate_learner_report_data(sv=100000, commission=9.95, impact=0.005):
    symbol = "JPM"
    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)
    learner = StrategyLearner(commission=commission, impact=impact, verbose=False)

    learner.add_evidence(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    sl_trades_is = learner.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    bm_trades_is = create_benchmark_trades(symbol=symbol, index = sl_trades_is.index)
    sl_portvals_is = msc.compute_portvals(sl_trades_is, sv, commission, impact)
    bm_portvals_is = msc.compute_portvals(bm_trades_is, sv, commission, impact)
    df_normed_is = pd.DataFrame({
        "Strategy Learner": sl_portvals_is / sl_portvals_is.iloc[0],
        "Benchmark": bm_portvals_is / bm_portvals_is.iloc[0]
    })
    plot_comparison(df_normed_is, f"Strategy Learner vs Benchmark ({symbol} In-Sample)", "images/learner_vs_benchmark_in_sample.png")

    sl_trades_oos = learner.testPolicy(symbol=symbol, sd=out_sample_sd, ed=out_sample_ed, sv=sv)
    bm_trades_oos = create_benchmark_trades(symbol=symbol, index = sl_trades_oos.index)
    sl_portvals_oos = msc.compute_portvals(sl_trades_oos, sv, commission, impact)
    bm_portvals_oos = msc.compute_portvals(bm_trades_oos, sv, commission, impact)
    df_normed_oos = pd.DataFrame({
        "Strategy Learner": sl_portvals_oos / sl_portvals_oos.iloc[0],
        "Benchmark": bm_portvals_oos / bm_portvals_oos.iloc[0]
    })
    plot_comparison(df_normed_oos, f"Strategy Learner vs Benchmark ({symbol} Out-of-Sample)", "images/learner_vs_benchmark_out_of_sample.png")


if __name__ == "__main__":
    sv_standard = 100000
    commission_standard = 9.95
    impact_standard = 0.005

    generate_manual_report_data(sv=sv_standard, commission=commission_standard, impact=impact_standard)
    generate_learner_report_data(sv=sv_standard, commission=commission_standard, impact=impact_standard)
    run_experiment1(sv=sv_standard, commission=commission_standard, impact=impact_standard)
    run_experiment2(sv=sv_standard)