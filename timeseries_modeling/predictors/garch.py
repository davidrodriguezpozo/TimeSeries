from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..utils import _get_col


class GARCHModel:
    
    @classmethod
    def local_variance(cls: GARCHModel, series: np.ndarray) -> np.ndarray:
        var = np.zeros(len(series))
        for i in range(5, len(series)- 5):
            new_var = np.var(series[i-5:i+5])
            var[i] = new_var
        
        for i in range(len(series) - 5, len(series)):
            var[i] = new_var
        return var
    
    @classmethod
    def compute_arch_predicted_error(cls: GARCHModel, series: pd.DataFrame, col: str = None, ar_deg: int = 4, arch_deg: int = 4) -> pd.DataFrame:
        series = series.copy()
        col = _get_col(series, col)
        predicted = sm.tsa.AutoReg(series[col].values, lags=ar_deg, trend='n').fit().predict()
        series["Predicted"] = predicted
        error_sq = (series["Predicted"] - series[col]) ** 2
        error_sq[error_sq.isna()] = 0
        error_predicted = sm.tsa.AutoReg(error_sq.values, lags=arch_deg, trend='c').fit().predict()
        series["Error Predicted"] = error_predicted
        series["Error"] = error_sq
        return series

    @classmethod
    def get_lower_upper_std(cls: GARCHModel, series: pd.DataFrame, col: str = None, ar_deg: int = 4, arch_deg: int = 4) -> pd.DataFrame:
        col = _get_col(series, col)
        series = cls.compute_predicted_error_df(series, col, ar_deg, arch_deg)
        error_prediction = np.sqrt(series["Error Predicted"].values)
        series["Lower"] = series[col] - error_prediction
        series["Upper"] = series[col] + error_prediction
        return series

    @classmethod
    def get_percentages(cls: GARCHModel, series: pd.DataFrame, col: str = None, ar_deg: int = 4, arch_deg: int = 4) -> Tuple[float, float]:
        col = _get_col(series, col)
        series = cls.get_lower_upper_std(series, col, ar_deg, arch_deg)
        error_prediction = np.sqrt(series["Error Predicted"].values)
        in_one_sd = (series['Upper'] > series[col]) & (series["Lower"] < series[col])
        in_two_sd = (series['Predicted'] + 2 * error_prediction > series[col]) & (series['Predicted'] - 2 * error_prediction < series[col])
        return in_one_sd.mean(), in_two_sd.mean()

    @classmethod
    def compute_predicted_error(cls: GARCHModel, series: pd.DataFrame, col: str = None, ar_deg: int = 4, arch_deg: int = 4) -> pd.DataFrame:
        series = series.copy()
        col = _get_col(series, col)
        predicted = sm.tsa.AutoReg(series[col].values, lags=ar_deg, trend='n').fit().predict()
        series["Predicted"] = predicted
        error_sq = (series["Predicted"] - series[col]) ** 2
        error_sq[error_sq.isna()] = 0
        error_predicted = sm.tsa.arima.ARIMA(error_sq.values, order=(arch_deg, 0, 3), trend='c').fit().predict()
        series["Error Predicted"] = error_predicted
        series["Error"] = error_sq
        return series
