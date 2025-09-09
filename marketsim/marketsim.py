""""""  		  	   		 	 	 			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
import os
import numpy as np			  	   		 	 	 			  		 			     			  	  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data  		  	   		 	 	 			  		 			     			  	 


def author():
    return "yhern3"  		 			     			  	 

def study_group():
    return "yhern3"

def get_adjusted_closing_prices(symbols, start_date, end_date):
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    return prices

def execute_order(assets, order_type, stock, num_stocks, total_amount):

    if order_type == 'SELL':
        if assets.get(stock) is None:
            assets[stock] = -num_stocks # this means shorting
        else:
            assets[stock] -= num_stocks # this means selling

    elif order_type == 'BUY':
        if assets.get(stock) is None:
            assets[stock] = num_stocks
        else:
            assets[stock] += num_stocks
        
    assets['cash'] += total_amount
    return assets

def calculate_portfolio_daily_val(assets, adj_closing_prices, date):
    stocks = [i for i in list(assets.keys()) if i != 'cash']
    sub = adj_closing_prices.loc[date]
    total_value = sum([sub[stock] for stock in stocks])

    return total_value + assets.get('cash')


def calculate_cash_change(order_type, share, adj_closing_price):
        return - share* adj_closing_price if order_type == 'BUY' else share * adj_closing_price

def execute_orders(order_df, adj_closing_prices, assets, commission, impact):
    prev_date = None

    for date in adj_closing_prices.index:
        if prev_date is not None:
            assets.loc[date] = assets.loc[prev_date]

        if date in order_df.index:
            for _, entry in order_df.loc[[date]].iterrows():
                symbol, order, shares = entry[['Symbol', 'Order', 'Shares']]
                price = adj_closing_prices.loc[date, symbol]            

                # we first calculate the new market price by considering impact, THEN we charge commission
                if order == 'BUY':
                    trade_price = price * (1 + impact)
                    total_cost = shares * trade_price + commission
                    assets.loc[date, symbol] += shares
                    assets.loc[date, 'cash'] -= total_cost

                elif order == 'SELL':
                    trade_price = price * (1 - impact)
                    total_revenue = shares * trade_price - commission
                    assets.loc[date, symbol] -= shares
                    assets.loc[date, 'cash'] += total_revenue

        prev_date = date

    assets_stocks = assets.drop(columns=['cash'])
    aligned_stocks, aligned_prices = assets_stocks.align(adj_closing_prices, join="inner", axis=1)
    assets_portfolio_value = aligned_stocks * aligned_prices
    assets_portfolio_value['cash'] = assets['cash']

    return assets_portfolio_value.sum(axis=1)


def compute_portvals(  		  	   		 	 	 			  		 			     			  	 
    orders_file="./orders/orders.csv",  		  	   		 	 	 			  		 			     			  	 
    start_val=1000000,  		  	   		 	 	 			  		 			     			  	 
    commission=9.95,  		  	   		 	 	 			  		 			     			  	 
    impact=0.005,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 

    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'], keep_date_col=True)
    orders.sort_index(inplace=True)

    adjusted_closing_prices = get_adjusted_closing_prices(list(orders['Symbol'].unique()), 
                                                            orders.index[0], 
                                                            orders.index[-1])

    assets = {"cash": [start_val] + [None] * (len(adjusted_closing_prices) - 1)}

    for stock in list(orders['Symbol'].unique()):
        assets[stock] = [0] + [None] * (len(adjusted_closing_prices) - 1)

    assets = pd.DataFrame(assets)
    assets.set_index(adjusted_closing_prices.index, inplace = True)

    daily_portfolio_value = execute_orders(orders, adjusted_closing_prices, assets, commission, impact)

    return daily_portfolio_value	
	 			     			  	  			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Helper function to test code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		 	 	 			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 	 	 			  		 			     			  	 
    # Define input parameters  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    of = "./orders/orders-02.csv"  		  	   		 	 	 			  		 			     			  	 
    sv = 1000000  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Process orders  		  	   		 	 	 			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	 	 			  		 			     			  	 
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 			  		 			     			  	 
    else:  		  	   		 	 	 			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Get portfolio stats  		  	   		 	 	 			  		 			     			  	 
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		 	 	 			  		 			     			  	 
    start_date = dt.datetime(2008, 1, 1)  		  	   		 	 	 			  		 			     			  	 
    end_date = dt.datetime(2008, 6, 1)  		  	   		 	 	 			  		 			     			  	 
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		 	 	 			  		 			     			  	 
        0.2,  		  	   		 	 	 			  		 			     			  	 
        0.01,  		  	   		 	 	 			  		 			     			  	 
        0.02,  		  	   		 	 	 			  		 			     			  	 
        1.5,  		  	   		 	 	 			  		 			     			  	 
    ]  		  	   		 	 	 			  		 			     			  	 
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		 	 	 			  		 			     			  	 
        0.2,  		  	   		 	 	 			  		 			     			  	 
        0.01,  		  	   		 	 	 			  		 			     			  	 
        0.02,  		  	   		 	 	 			  		 			     			  	 
        1.5,  		  	   		 	 	 			  		 			     			  	 
    ]  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Compare portfolio against $SPX  		  	   		 	 	 			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
