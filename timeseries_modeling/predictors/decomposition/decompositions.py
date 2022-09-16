from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

from ...smoothers import exponential
from ...utils import _create_diff_col, _get_col
from ..autoregression.v1 import AutoRegressionPredictor


class Decompositions:

    @classmethod
    def decompose_df(cls: Decompositions, 
                    series: pd.DataFrame, 
                    col: str = None, 
                    alpha: float = 0.5, 
                    as_values: bool = False) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """Decompose the given dataframe into smoothed and residual values using exponential smoothing

        Args:
            series (pd.DataFrame): DataFrame to handle
            col (str, optional): If DF has more than one col, specifiy it here. Defaults to None.
            alpha (int, optional): Alpha parameter for the exponential smoothing. Defaults to 0.5.
            as_values (bool, optional): If True, return values instead of a DF. Defaults to False.

        Returns:
            pd.DataFrame: The updaated dataframe with the decomposed columns
        """
        col = _get_col(series, col)
        smoothed = exponential(series, col, alpha, as_array=True)
        residual = series[col].values - smoothed
        if as_values:
            return smoothed, residual
        series = series.copy()
        series["Smoothed"] = smoothed
        series["Residual"] = residual
        series["Res diff"] = series["Residual"] - series["Residual"].shift()
        return series

    @classmethod
    def predict(cls: Decompositions, series: pd.DataFrame, col: str = None) -> pd.DataFrame:
        """Append a new column 'Doc. prediction {col}' to the DF with a decomposition prediction
        using exponential smoothing, AR for the residual, and naive prediction for the trend. 

        Args:
            series (pd.DataFrame): DataFrame to handle. 
            col (str, optional): Name of the column (if more than one in DF). Defaults to None.

        Returns:
            pd.DataFrame: DF with the added 'Doc. prediction {col}' column
        """
        series = series.copy()
        col = _get_col(series, col)
        decomposed_df = cls.decompose_df(series, col).dropna()
        residue_predicted_values = AutoRegressionPredictor.predict_ar(decomposed_df["Res diff"].values, 12)
        residue_prediction = residue_predicted_values + decomposed_df["Residual"].shift()
        trend_differences = _create_diff_col(decomposed_df, col, as_values=True, lag=2)
        trend_prediction = decomposed_df[col].shift(1) + trend_differences
        series[f"Dec. prediction {col}"] = trend_prediction
        return series


    @classmethod
    def predict_lowess(cls: Decompositions, series: pd.DataFrame, col: str = None) -> pd.DataFrame:
        """Generate a prediction column using lowess decomposition

        Args:
            series (pd.DataFrame): The DF to handle
            col (str, optional): If DF has more than one column, specify it here. Defaults to None.

        Returns:
            pd.DataFrame: A new DF with a new column 'Lowess prediction {col}'
        """
        series = series.copy()
        col = _get_col(series, col)
        series["smoothed"] = sm.nonparametric.lowess(series["value"], series.index, frac=0.03)[:,1]
        residue = series["value"] - series["smoothed"]
        res_prediction = AutoRegressionPredictor.predict_ar(residue, 12)
        trend_diff = _create_diff_col(series, col="value", as_values=True, lag=2)
        trend_prediction = series["smoothed"].shift() + trend_diff
        series[f"Lowess prediction {col}"] = trend_prediction
        return series.drop(columns="smoothed")

    @classmethod
    def predict_STL(cls: Decompositions, series: pd.DataFrame, col: str = None, period: int = None) -> pd.DataFrame:
        col = _get_col(series, col)
        series = series.copy().dropna()
        decomp = STL(series.dropna()[col].values, period=period).fit()
        residual = decomp.resid
        seasonal = decomp.seasonal
        trend = decomp.trend
        seasonal = pd.Series(seasonal)
        trends_df = pd.DataFrame({"value": trend})
        trend_diff = _create_diff_col(trends_df, col="value", as_values=True, lag=2)
        trend_prediction = trends_df["value"].shift() + trend_diff
        seasonal_predicted = seasonal.shift(12)
        res_prediction = AutoRegressionPredictor.predict_ar(residual, 12)
        
        prediction = trend_prediction + seasonal_predicted + res_prediction
        series["Prediction STL"] = prediction.values
        return series
