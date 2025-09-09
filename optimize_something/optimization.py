""""""  		  	   		 	 	 			  		 			     			  	 
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import matplotlib.pyplot as plt  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data  		  	   		 	 	 			  		 			     			  	 
import scipy.optimize as spo
  		  	   		 	 	 			  		 			     			  	 
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			     			  	 
# The student must update this code to properly implement the functionality  	

def author():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: str  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return "yhern3"  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def gtid():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT ID of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: int  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return 903870865	

def study_group():
    return "Yulian Lee Ying Hern"	  	   		 	 	 			  		 			     			  	 

def calculate_portfolio_daily_value(prices: pd.DataFrame, allocations: list = None):
    """
    Calculate daily value of portfolio. 
    Returns the normalized daily value if there is only one stock.
    Return the normalized daily value multiplied by the allocations if there are multiple stocks in the portfolio. 
    """
    if isinstance(prices, pd.DataFrame):
        start_val = prices.iloc[0,:]
    elif isinstance(prices, pd.Series):
        start_val = prices.iloc[0]

    normalized_prices = prices/start_val

    if allocations is not None:
        allocated_values = normalized_prices.multiply(allocations)
        total_daily_value = allocated_values.sum(axis=1)

        return total_daily_value
    else:
        return normalized_prices
    

def calculate_negative_sharpe_ratio_allocations(allocs, daily_prices):
    "Minimize the negative value of the sharpe ratio, which is the same as maximizing the sharpe ratio."
    portfolio_daily_values = calculate_portfolio_daily_value(daily_prices, allocs)
    daily_returns = (portfolio_daily_values / portfolio_daily_values.shift(1)) - 1
    daily_returns = daily_returns[1:]
    return -(np.mean(daily_returns)/np.std(daily_returns, ddof=1))

def calculate_portfolio_stats(daily_prices: pd.DataFrame, allocs:list):
    portfolio_daily_values = calculate_portfolio_daily_value(daily_prices, allocs)

    # calculate daily returns
    # we use daily values instead of price since we do not need to specify a starting portfolio value
    daily_returns = (portfolio_daily_values / portfolio_daily_values.shift(1)) - 1
    daily_returns = daily_returns[1:]

    # calculate stats
    avg_daily_returns = np.mean(daily_returns)
    cumulative_returns = (portfolio_daily_values.iloc[-1] / portfolio_daily_values.iloc[0]) - 1
    std_dev_daily_returns = np.std(daily_returns, ddof=1)
    sharpe_ratio = (avg_daily_returns) / std_dev_daily_returns

    return cumulative_returns, avg_daily_returns, std_dev_daily_returns, sharpe_ratio

def calculate_optimal_allocation(daily_prices, num_stocks):

    initial_guess = np.float32([1/num_stocks]* num_stocks) 
    constraints = {'type': 'eq', 'fun': lambda allocs: np.sum(allocs) - 1}
    bounds = [(0, 1)] * num_stocks
    result = spo.minimize(calculate_negative_sharpe_ratio_allocations,
                          initial_guess, 
                          args=(daily_prices,), 
                          method = 'SLSQP', 
                          bounds = bounds,
                          constraints = constraints,
                          options={'disp': True}
                          )
    
    return result.x

def optimize_portfolio(  		  	   		 	 	 			  		 			     			  	 
    sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
    syms=["GOOG", "AAPL", "GLD", "XOM"],	  	   		 	 	 			  		 			     			  	 
    gen_plot=False,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 			  		 			     			  	 
    statistics.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
    :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
    :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 			  		 			     			  	 
        symbol in the data directory)  		  	   		 	 	 			  		 			     			  	 
    :type syms: list  		  	   		 	 	 			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			     			  	 
        code with gen_plot = False.  		  	   		 	 	 			  		 			     			  	 
    :type gen_plot: bool  		  	   		 	 	 			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 			  		 			     			  	 
    :rtype: tuple  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	 	 			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		   	 	 			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		

    prices_all.fillna(method = 'ffill', inplace  = True) # Fill forward
    prices_all.fillna(method = 'bfill', inplace = True) # Fill backward

    prices = prices_all[syms]  # only portfolio symbols  		# index = date, columns = symbols

    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			     			  	 
    # find the allocations for the optimal portfolio  		  	   		 	 	 			  		 			     			  	 
    # note that the values here ARE NOT meant to be correct for a test case  	
    allocs = calculate_optimal_allocation(prices, len(syms)) # add code here to find the allocations   		

    cr, adr, sddr, sr = calculate_portfolio_stats(prices, allocs) # add code here to compute stats    		 			     			  	 
    
    # Get daily portfolio value  	   		 	 	 			  		 			     			  	 
    port_daily_val = calculate_portfolio_daily_value(prices, allocs)  # add code here to compute daily portfolio values  		  	   		 	 	 			  		 			     			  	 
    spy_port_daily_val = calculate_portfolio_daily_value(prices_SPY)

    # Compare daily portfolio value with SPY using a normalized plot  		
    # Normalize the SPY value and the daily value from the portfolio by dividing by the first value then plot  	   		 	 	 			  		 			     			  	 
    if gen_plot:
        # port_val_normalized = port_val / port_val.iloc[0]
        # prices_SPY_normalized = prices_SPY / prices_SPY.iloc[0]
        
        df_temp = pd.concat(
            [port_daily_val, spy_port_daily_val],
            keys=["Portfolio", "SPY"],
            axis=1
        )
        
        df_temp.plot(title="Daily Portfolio Value vs. Daily Value SPY", figsize=(10, 6))
        plt.grid(color = 'black', linestyle = 'dotted')
        plt.xlabel("Date")
        plt.ylabel("Normalized Value")
        plt.savefig("images/plot.png")	 
        plt.close()	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    return allocs, cr, adr, sddr, sr  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    start_date = dt.datetime(2009, 1, 1)  		  	   		 	 	 			  		 			     			  	 
    end_date = dt.datetime(2010, 1, 1)  		  	   		 	 	 			  		 			     			  	 
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Assess the portfolio  		  	   		 	 	 			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	 	 			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True		  	   		 	 	 			  		 			     			  	 
    )  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Print statistics  		  	   		 	 	 			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		 	 	 			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return: {cr}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 			  		 			     			  	 
    # Do not assume that it will be called  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 