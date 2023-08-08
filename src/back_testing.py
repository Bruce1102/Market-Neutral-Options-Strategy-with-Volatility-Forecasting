import numpy as np
import pandas as pd
from typing import Union, Callable, List, Tuple, Optional
from src.options import OptionCombination


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


def back_testing_strategy(df: pd.DataFrame, 
                          strategy: Callable[..., Optional[OptionCombination]], 
                          forecast_volatility: Callable[..., float], 
                          n_days: int=7,
                          stop_loss: Optional[Callable[..., int]]=None,
                          vol_lookback: int=5,
                          confidence: float=1.96,
                          T: float=7/365,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes the provided strategy on the dataframe using historical data.

    Args:
        df (pd.DataFrame): Dataframe with columns 'SPX_Close' and 'IRX_Close'.
        strategy (Callable): Strategy function to execute.
        forecast_volatility (Callable): Function to forecast volatility.
        stop_loss (Optional[Callable], default=None): Optional stop loss function. If provided, 
            checks when the trade should be exited before expiry.

    Returns:
        Tuple[List[float], List[float], List[float]]: Lists containing pnl, risk-free rate, 
                                                     and forecasted volatility.
    """
    DAYS_PER_WEEK = 5

    # Validate input
    if df.empty:
        raise ValueError("The input dataframe is empty.")
    if 'SPX_Close' not in df.columns or 'IRX_Close' not in df.columns:
        raise ValueError("Expected columns 'SPX_Close' and 'IRX_Close' in the dataframe.")

    trade = None 
    initial_value = []
    pnl = []
    rf  = []
    for_vol = []
    date = []

    for i in range(0, int(len(df)/DAYS_PER_WEEK)):
        idx = i * DAYS_PER_WEEK 
        last_price = df["SPX_Close"].iloc[idx + DAYS_PER_WEEK - 1]
        last_ir    = df["IRX_Close"].iloc[idx + DAYS_PER_WEEK - 1]
        vol_forecast_periods = (max(0, idx + DAYS_PER_WEEK - vol_lookback), idx + DAYS_PER_WEEK)

        # Check existing trade and compute pnl if necessary
        if trade:
            if stop_loss:
                # Compute break even points
                break_even_idx = stop_loss(trade, df["SPX_Close"].iloc[idx:idx+DAYS_PER_WEEK].values)

                if break_even_idx != DAYS_PER_WEEK-1:
                    cur_val = trade.current_value(T - break_even_idx/365, df["SPX_Close"].iloc[idx+break_even_idx])
                    pnl_temp = cur_val - trade.get_initial_value()
                else:
                    pnl_temp = trade.pnl_at_expiry(last_price)
            else:
                pnl_temp = trade.pnl_at_expiry(last_price)

            initial_value.append(trade.get_initial_value())
            pnl.append(pnl_temp)
            rf.append(last_ir)
            date.append(df.index[idx-1])
            trade = None

        # Forecast volatility
        forecasted_vol = forecast_volatility(df.iloc[vol_forecast_periods[0]:vol_forecast_periods[1]], n_days)


        # Execute strategy and set trade for next iteration
        if not np.isnan(forecasted_vol):
            trade = strategy(df.iloc[:idx + DAYS_PER_WEEK], forecasted_vol, confidence, long_short='short', T=T)

    # return np.array(initial_value), np.array(pnl), np.array(rf), np.array(for_vol), np.array(date)
    return np.array(initial_value), np.array(pnl), np.array(rf), np.array(date)

def sharpe_ratio(pnl: Union[np.array, List[float]], 
                 risk_free: Union[np.array, List[float]]) -> float:
    """
    Compute the Sharpe Ratio given arrays/lists of PnL values and risk-free rates.

    Args:
        pnl (Union[np.array, List[float]]): Array or list of PnL values for each period.
        risk_free (Union[np.array, List[float]]): Array or list of risk-free rates for each period.

    Returns:
        float: The computed Sharpe Ratio.

    Raises:
        ValueError: If the lengths of pnl and risk_free are not the same.
        ZeroDivisionError: If the standard deviation of pnl values is zero.
    """
    
    # Ensure pnl and risk_free have the same length
    if len(pnl) != len(risk_free):
        raise ValueError("The lengths of pnl and risk_free must be the same.")

    # Convert to numpy arrays for more efficient calculations
    pnl = np.array(pnl)
    risk_free = np.array(risk_free)
    
    # Calculate expected portfolio return and average risk-free rate
    expected_return = pnl.mean()
    avg_risk_free_rate = risk_free.mean()
    
    # Calculate portfolio standard deviation
    std_dev = pnl.std()
    
    # Check for zero standard deviation
    if std_dev == 0:
        raise ZeroDivisionError("Standard deviation of pnl values is zero, can't compute Sharpe Ratio.")
    
    # Calculate Sharpe Ratio
    sharpe = (expected_return - avg_risk_free_rate) / std_dev
    
    return sharpe



def information_ratio(strategy_returns: list, benchmark_returns: list) -> float:
    """
    Calculates the Information Ratio.

    Parameters:
    - strategy_returns (list): List of portfolio returns (in percentages).
    - benchmark_returns (list): List of benchmark returns, e.g., S&P 500 (in percentages).

    Returns:
    - float: Information Ratio.
    """
    
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("The lengths of strategy_returns and benchmark_returns must be the same.")
    
    # Calculate active returns (difference between strategy returns and benchmark returns)
    active_returns = [strategy - benchmark for strategy, benchmark in zip(strategy_returns, benchmark_returns)]
    
    # Calculate average active return and tracking error
    avg_active_return = sum(active_returns) / len(active_returns)
    tracking_error = (sum([(ret - avg_active_return) ** 2 for ret in active_returns]) / len(active_returns)) ** 0.5
    
    # Calculate Information Ratio
    if tracking_error == 0:  # Avoid division by zero
        return float('inf') if avg_active_return > 0 else float('-inf')
    else:
        return avg_active_return / tracking_error