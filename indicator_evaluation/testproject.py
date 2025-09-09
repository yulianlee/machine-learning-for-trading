import pandas as pd
import datetime as dt
from indicators import main
from TheoreticallyOptimalStrategy import run_optimal
import matplotlib.pyplot as plt

def author():
    return "yhern3" 

def study_group():
    return "yhern3"

def run(symbol = 'JPM', 
        start_date = dt.datetime(2008, 1, 1), 
        end_date = dt.datetime(2009, 12, 31),
        start_val = 100000):
    main(symbol, start_date, end_date, start_val)
    run_optimal(symbol, start_date, end_date, start_val) # run_optimal calls testPolicy to generate optimal strategy


if __name__ == '__main__':
    run()


