from numpy import log
from math import sqrt, exp
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import date
import yfinance as yf
from sys import argv

# returns d1, d2 for Black-Scholes
def d(P, X, T, r, s):
    d1 = ( log(P/X) + (r + s**2 / 2)*T ) / (s * sqrt(T))
    d2 = d1 - s * sqrt(T)
    return d1, d2

# returns Black-Scholes call price
def call(P, X, T, r, s):
    d1, d2 = d(P, X, T, r, s)
    C = P*norm.cdf(d1) - X*exp(-r*T)*norm.cdf(d2)
    return C

# uses iterative guessing i.e. Newton's method to estimate implied volatility
def call_iv(P, X, T, r, C):
    # guesses for volatility - we run Newton's method on multiple guesses to increase likelihood of convergence
    s_list = [x * 0.005 for x in range(10, 300, 5)]
    
    for s in s_list:
        # defining error and bounds - one for convergence and divergence
        err = 1
        good_bound = 0.000001
        bad_bound = 100
        success = True
        while (err >= good_bound):
            if (err >= bad_bound or s <= 0):
                success = False
                break

            diff = call(P, X, T, r, s) - C
            d1, d2 = d(P, X, T, r, s)
            vega = P * norm.pdf(d1)*sqrt(T)
            if vega < 0.001:
                success = False
                break
            s = s - (diff / vega)
            err = abs(diff)
        
        if success:
            return s
    
    return 0

def plot_ivs(symbol):
    # risk-free rate; https://ycharts.com/indicators/10_year_treasury_rate
    r = 0.0285

    # source stock data from Yahoo Finance using yfinance library
    ticker = yf.Ticker(symbol)

    if not ticker.options:
        return 1

    expirys = []
    moneynesses = []
    ivs = []

    today = date.today()

    for d in ticker.options:
        # Find time to expiry date
        year, month, day = d.split("-")
        date_parsed = date(int(year), int(month), int(day))
        T = (date_parsed - today).days / 365.0

        if T == 0:
            continue
        
        calls = ticker.option_chain(d)[0]
        price = float(ticker.history(d).drop_duplicates(subset=['Close'])['Close'])

        for _, call in calls.iterrows():
            iv = call_iv(price, call['strike'], T, r, call['lastPrice'])
            moneyness = call['strike'] / price

            # calc_iv == 0 implies that the call_iv failed to converge to a positive volatility
            # checks for reasonable bounds on values
            if iv != 0 and 0 < moneyness and moneyness < 2:
                expirys.append(T)
                moneynesses.append(moneyness)
                ivs.append(iv)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(moneynesses, expirys, ivs, c=ivs, cmap='viridis')
    ax.set_title(f"Implied Volatility for { symbol } Call Options, { today }")
    ax.set_xlabel('moneyness', fontsize=10)
    ax.set_ylabel('time to expiry', fontsize=10)
    ax.set_zlabel('implied volatility', fontsize=10)
    plt.show()

    return 0

def main():
    if len(argv) < 2:
        print("Provide ticker symbol.")
        return
    
    if plot_ivs(argv[1]) != 0:
        print("Invalid ticker symbol.")

if __name__ == "__main__":
    main()