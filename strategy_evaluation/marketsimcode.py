import pandas as pd
import numpy as np
import datetime as dt
import util as ut

def author():
    return 'yhern3'

def study_group():
    return 'yhern3'

def compute_portvals(orders_df, start_val=100000, commission=0.0, impact=0.0):
    symbols = orders_df.columns.tolist()
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    prices_all = ut.get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices_all[symbols]
    prices = prices.ffill().bfill()
    prices["CASH"] = 1.0
    trades = pd.DataFrame(0.0, index=prices.index, columns=symbols + ["CASH"])
    
    for date, order_row in orders_df.iterrows():
        for symbol in symbols:
            order_shares = order_row[symbol]

            if order_shares == 0:
                continue
            price = prices.loc[date, symbol]
            cost = price * order_shares
            trade_cost = cost + commission + (abs(cost) * impact)
            trades.loc[date, symbol] += order_shares
            trades.loc[date, "CASH"] -= trade_cost
    
    holdings = trades.copy()
    holdings.loc[holdings.index[0], "CASH"] = start_val + holdings.loc[holdings.index[0], "CASH"]
    holdings = holdings.cumsum(axis=0)
    values = holdings * prices
    portvals = values.sum(axis=1)
    
    return portvals