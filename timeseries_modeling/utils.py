import math
from typing import Optional

import numpy as np
import pandas as pd


def _get_col(df: pd.DataFrame, col: Optional[str]) -> str:
        """Get name of the column of the given DataFrame

        Args:
            df (pd.DataFrame): DataFrame
            col (Optional[str]): Name of column (if more than one)

        Raises:
            ValueError: If the number of columns is more than one, and no name is provided

        Returns:
            str: Name of the column
        """
        if len(df.columns) > 1 and not col:
            raise ValueError("Number of columns must be 1. Not a Time Series")
        return col or df.columns[0]


def _plot_dataframe(df: pd.DataFrame, title: str = None) -> None:
    df.plot(title=title, ylabel=df.columns.to_list()[0])

def _create_diff_col(df: pd.DataFrame, col: str, as_values: bool = False, lag: int = 1) -> pd.DataFrame:
    """Create a `diff` column in the dataframe, that consists on the sdubtraction of lagged values

    Args:
        df (pd.DataFrame): DataFrame that contains the information
        col (str): Name of the column
        as_values (bool, optional): If True, return np.ndarray instead of appending column. Defaults to False.
        lag (int, optional): Number of rows to lag. Defaults to 1.

    Returns:
        pd.DataFrame: A new DataFrame with the added column
    """
    differences = df[col].shift(lag-1) - df[col].shift(lag)
    if as_values:
        return differences.values
    df = df.copy()
    df["diff"] = differences
    return df

def adjust_trend(time_series: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """
    This method removes the trend of the input time series. If a Time Series
    has an upward or downard trend, the result of this method will return
    a dataframe with a new column "diff", which is the difference between the
    value column of the current timestamp, and the previous timestamp
    """
    df = time_series.copy()
    col = _get_col(df, col)
    df = _create_diff_col(df, col)
    df.plot()
    return df

def adjust_season(time_series: pd.DataFrame, period: int, col: str = None) -> pd.DataFrame:
    """
    This method adjusts the time series' seasonlity factor and removes it in 
    order to see what is the real trend and remove the noise of the time series.
    The period is always the number of *months*. All the dates will be converted to 
    months before applying the regularisation.
    """
    df = time_series.copy()
    col = _get_col(df, col)
    season_sum = [0] * period
    season_count = [0] * period
    df["Month"] = df.index.month
    for _, row in df.iterrows():
        season_sum[int(row["Month"]) % period] += row[col]
        season_count[int(row["Month"]) % period] += 1
    season_means = [season_sum[i]/season_count[i] for i in range(period)]
    total_mean = df[col].mean(axis=0)
    season_offsets = [season_means[i] - total_mean for i in range(period)]
    adjusted = list()
    for _, row in df.iterrows():
        month = int(row["Month"])
        adjusted.append(row[col] - season_offsets[month % period])
    df[f"{col}_sa"] = adjusted
    df.drop(columns=["Month"], inplace=True)
    df.plot()
    return df

def compute_log_df( time_series: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """Compute the log of the given dataframe as an extra column `Log {col}` and the differences
    of with respect to the previous value in another column `Diff Log {col}`

    Args:
        time_series (pd.DataFrame): The dataframe we want to compute the log of
        col (str, optional): If the DataFrame has more than one col, specify here which one we want to work with. Defaults to None.

    Returns:
        pd.DataFrame: The same dataframe but with the added columns
    """
    df = time_series.copy()
    col = _get_col(df, col)
    df[f"Log {col}"] = np.log(df[col])
    df[f"Diff Log {col}"] = np.abs(df[f"Log {col}"] - df[f"Log {col}"].shift())
    df = df.drop(columns=[col])
    return df


def get_correlation(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xs_diffs = xs - np.mean(xs)
    ys_diffs = ys - np.mean(ys)

    return np.dot(xs_diffs, ys_diffs)/  np.sqrt( (np.dot(xs_diffs, xs_diffs) * np.dot(ys_diffs, ys_diffs)))


def rmse(predicted: pd.Series, actual: pd.Series) -> float:
    diffs = predicted.reset_index(drop=True) - actual.reset_index(drop=True)
    diffs[diffs.isna()] = 0
    return math.sqrt(np.mean(diffs ** 2))
