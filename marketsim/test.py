import pandas as pd
import datetime as dt
from util import get_data, plot_data  


def get_adjusted_closing_prices(symbols, start_date, end_date):
    """
    Start date and end date are the index.
    
    Params
    symbols - List of strings like: ["AAPL", "GOOG", "AMZN"]
    start_date - string in YYYY-MM-DD format
    end_date - string in YYYY-MM-DD format

    Returns
    Dataframe containing adjusted closing prices per symbol. 
    One column = one symbol and their adjusted closing prices for that day
    Index = Date range between start and end date
    """

    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    return prices

orders = pd.read_csv("./orders/orders-01.csv", index_col='Date', parse_dates=True, na_values=['nan'], keep_date_col=True)
orders.sort_index(inplace=True)
orders.reset_index(inplace = True)

orders.set_index(['Date', 'Symbol'], inplace = True)

print(orders.head())

print(orders.loc['2011-01-10', 'AAPL'])


# adjusted_closing_prices = get_adjusted_closing_prices(list(orders['Symbol'].unique()), 
#                                                         orders.index[0], 
#                                                         orders.index[-1])

# print(adjusted_closing_prices.head())

# for _, a in adjusted_closing_prices.head().iterrows():
#     print(a)
#     print('HERE', )
# print(adjusted_closing_prices.loc['2011-01-10']['SPY'])




# # print(adjusted_closing_prices.head())
# orders = orders.join(adjusted_closing_prices)
# print(orders.head())

# orders["adj_closing_price"] = orders.apply(lambda row: row[row["Symbol"]], axis=1)
# orders = orders[['Symbol', 'Order', 'Shares', 'adj_closing_price']]
# print(orders.head())

### FOR TESTING THE get_data HELPER FUNCTION

# def get_stock_prices_with_cash(symbols, start_date, end_date):
#     """Start date and end date are the index"""
#     prices = get_data(symbols, pd.date_range(start_date, end_date))
#     prices.fillna(method='ffill', inplace=True)
#     prices.fillna(method='bfill', inplace=True)
#     prices = prices[symbols]
#     return prices

# start_date = "2012-08-30"
# end_date = "2012-09-12"
# dates = pd.date_range(start=start_date, end=end_date)

# # List of symbols
# symbols = ["AAPL", "GOOG", "AMZN"]

# # Call the function
# df = get_data(symbols, dates)

# print(df.head())
