import numpy as np
from scipy.stats import norm

def annualized_volatility(prices: np.array) -> float:
    """
    Calculate the annualized volatility (in decimal form) from a numpy array of prices.

    Parameters:
    - prices: A numpy array of prices.

    Returns:
    - Annualized volatility in decimal form.
    """
    # Calculate daily returns in decimal form
    daily_returns = np.diff(prices) / prices[:-1]

    # Calculate the standard deviation of daily returns
    daily_std = np.std(daily_returns)

    # Annualize the daily standard deviation
    annualized_std = daily_std * (252**0.5)

    return annualized_std


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.

    Parameters:
    - S: Current stock price
    - K: Option strike price
    - T: Time to option expiry (in years)
    - r: Risk-free rate (annualized)
    - sigma: Volatility of the underlying stock (annualized)
    - option_type: 'call' for call option, 'put' for put option

    Returns:
    - Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'")
    
    return option_price


def straddle_price(S, K, T, r, sigma):
    """
    Calculate the price of a straddle using black scholes formula.

    Parameters:
    - S: Current stock price
    - K: Option strike price
    - T: Time to option expiry (in years)
    - r: Risk-free rate (annualized)
    - sigma: Volatility of the underlying stock (annualized)

    Returns:
    - Straddle options price
    """
    call_price = black_scholes(S, K, T, r, sigma, option_type='call')
    put_price  = black_scholes(S, K, T, r, sigma, option_type='put')
    return  call_price + put_price

