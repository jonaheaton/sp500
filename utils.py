import yfinance as yf
import pandas as pd
import os
import numpy as np
import requests

from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_table(filename):

    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col='date')
        return df


def compute_gini(x, w=None):
    # https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n



def get_previous_month(date_str):
    date_format = "%Y-%m-%d"
    date_obj = datetime.strptime(date_str, date_format)
    previous_month = date_obj - relativedelta(months=1)
    return previous_month.strftime(date_format)

def get_market_cap(ticker, date):
    try:
        stock = yf.Ticker(ticker)
        start_date = get_previous_month(date)
        history = stock.history(start=start_date, end=date)
        if history.empty:
            print(f"No data available for {ticker} on {date}")
            return None
        close_price = history['Close'][-1]
        shares_outstanding = stock.info.get('sharesOutstanding')
        if shares_outstanding is None:
            print(f"No data available for shares outstanding of {ticker}")
            return None
        market_cap = close_price * shares_outstanding
        return market_cap
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    

# Using Polygon to get stock data
def get_polygon_data(ticker, snap_shot):
    # polygon is more reliable than yfinance, but it requires an API key

    apikey = 'nsyGfUpE71vyC1YNfUJnyjQl7P2Cgd2L'
    url = f'https://api.polygon.io/v3/reference/tickers/{ticker}?date={snap_shot}&apiKey={apikey}'

    r = requests.get(url)
    data = r.json()
    return data


def get_sp500_constituents(snap_shot_day,verbose=False):
    filename = 'S&P 500 Historical Components & Changes(12-30-2023).csv'
    df = get_table(filename)
    # Convert ticker column from csv to list, then sort.
    df['tickers'] = df['tickers'].apply(lambda x: sorted(x.split(',')))

    # Get the synbols on snap_shot date by filtering df by rows before or on the snap_shot date,
    # then picking the last row.
    df2 = df[df.index <= snap_shot_day]
    last_row = df2.tail(1)

    past = last_row['tickers'].iloc[0]
    if verbose:
        print('*'*40, f'S&P 500 on {snap_shot_day}', '*'*40)
        print(past)

    return past

def get_sp500_market_weights(snap_shot_day,verbose=False,stop_at=0):
    tickers = get_sp500_constituents(snap_shot_day,verbose)
    if stop_at>0:
        tickers = tickers[:stop_at]
    found_tickers = []
    missing_tickers = []
    if verbose:
        print(f'Number of tickers: {len(tickers)}')
    market_weights = []
    for ticker in tickers:
        market_cap = get_market_cap(ticker, snap_shot_day)
        if market_cap:
            market_weights.append(market_cap)
            found_tickers.append(ticker)
        else:
            missing_tickers.append(ticker)
    return market_weights, found_tickers, missing_tickers