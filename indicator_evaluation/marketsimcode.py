import pandas as pd
from util import get_data

def author():
    return "yhern3" 


def get_adjusted_closing_prices(symbols, start_date, end_date):
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    return prices

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
    trades: pd.DataFrame,		  	   		 	 	 			  		 			     			  	 
    start_val=1000000,  		  	   		 	 	 			  		 			     			  	 
    commission=9.95,  		  	   		 	 	 			  		 			     			  	 
    impact=0.005,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 

    # orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'], keep_date_col=True)
    # orders.sort_index(inplace=True)

    adjusted_closing_prices = get_adjusted_closing_prices(list(trades['Symbol'].unique()), 
                                                            trades.index[0], 
                                                            trades.index[-1])

    assets = {"cash": [start_val] + [None] * (len(adjusted_closing_prices) - 1)}

    for stock in list(trades['Symbol'].unique()):
        assets[stock] = [0] + [None] * (len(adjusted_closing_prices) - 1)

    assets = pd.DataFrame(assets)
    assets.set_index(adjusted_closing_prices.index, inplace = True)

    daily_portfolio_value = execute_orders(trades, adjusted_closing_prices, assets, commission, impact)

    return daily_portfolio_value	