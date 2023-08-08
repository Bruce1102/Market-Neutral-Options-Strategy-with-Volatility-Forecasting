import pandas as pd
import numpy as np
import datetime as dt

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and convert the index to datetime format.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    
    Returns:
    - pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path, index_col=0)
        data.index = pd.to_datetime(data.index)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def filter_data_by_year_range(data: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Filter data based on a range of years.
    
    Parameters:
    - data (pd.DataFrame): Input data.
    - start_year (int): Start year for filtering.
    - end_year (int): End year for filtering.
    
    Returns:
    - pd.DataFrame: Filtered data.
    """
    data["Year"] = data.index.year
    return data[(data["Year"] >= start_year) & (data["Year"] <= end_year)]

def append_string_to_elements(elements: list, suffix: str) -> list:
    """
    Append a string to each element in a list.
    
    Parameters:
    - elements (list): List of strings.
    - suffix (str): String to append.
    
    Returns:
    - list: List with appended strings.
    """
    return [element + suffix for element in elements]

def save_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a CSV file.
    
    Parameters:
    - data (pd.DataFrame): Data to save.
    - file_path (str): Path to the CSV file.
    """
    try:
        data.to_csv(file_path)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error: {e}")

def closest_date(target_date: dt.datetime, df: pd.DataFrame) -> (dt.datetime, dt.datetime):
    """
    Find the closest dates before and after the provided target date in the DataFrame's index.
    
    Parameters:
    - target_date (dt.datetime): The date for which neighboring dates are to be found.
    - df (pd.DataFrame): DataFrame with a datetime index to search for the date.
    
    Returns:
    - Tuple[dt.datetime, dt.datetime]: Neighboring dates before and after the target date.
    """
    # Ensure the DataFrame is sorted by the index
    df = df.sort_index()
    
    # Find dates before and after the target date
    before = df[df.index < target_date].index.max()
    after = df[df.index > target_date].index.min()
    
    return before, after

def compute_returns(arr: np.ndarray) -> float:
    """
    Compute the simple return over an array of prices.
    
    Parameters:
    - arr (np.ndarray): Array of prices.
    
    Returns:
    - float: Simple return.
    """
    if len(arr) < 2 or arr[0] == 0:
        raise ValueError("Array should have at least two elements and the first element should not be zero.")
    
    return (arr[-1] - arr[0]) / arr[0]

def compute_log_returns(arr: np.ndarray) -> float:
    """
    Compute the log return over an array of prices.
    
    Parameters:
    - arr (np.ndarray): Array of prices.
    
    Returns:
    - float: Log return.
    """
    if len(arr) < 2 or arr[0] <= 0 or arr[-1] <= 0:
        raise ValueError("Array should have at least two elements and both the first and last elements should be positive.")
    
    return np.log(arr[-1] / arr[0])
