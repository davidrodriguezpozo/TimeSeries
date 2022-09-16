
import numpy as np
import pandas as pd

from .utils import _get_col


def moving_average(series: pd.DataFrame, col: str = None, number: int = 3) -> pd.DataFrame:
        """Compute the smoothed moving average of the given time series 

        Args:
            series (pd.DataFrame): DataFrame that will be used to compute the MA
            col (str, optional): If the DF has more than one column, specify the column here. Defaults to None.
            number (int, optional): Number of points to compute the average with. Defaults to 3.

        Returns:
            pd.DataFrame: The DF with an added column
        """
        series = series.copy()
        col = _get_col(series, col)
        smoothed = series[col] - series[col]
        for i in range(number):
            smoothed = smoothed + series[col].shift(i)
        series[f"MA smooth {col} ({number})"] = smoothed/number
        return series


def exponential(series: pd.DataFrame, col:str = None, alpha: float = 1.5, as_array: bool = False) -> pd.DataFrame:
    """Performs an exponential smoothing of the given dataframe

    Args:
        series (pd.DataFrame): DataFrame that will be used to compute the MA
        col (str, optional): If the DF has more than one column, specify the column here. Defaults to None.
        number (int, optional): Number of points to compute the average with. Defaults to 3.

    Returns:
        pd.DataFrame: The result DF with the added column
    """
    series = series.copy()
    col = _get_col(series, col)
    smoothed_value = series[col][0]
    results = []
    for v in series[col]:
        smoothed_value = (1-alpha) * smoothed_value + alpha * v
        results.append(smoothed_value)
    
    if as_array:
        return np.array(results)

    series[f"Exp smooth {col} ({alpha})"] = results
    return series
