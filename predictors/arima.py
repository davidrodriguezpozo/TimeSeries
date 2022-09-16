from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..utils import _get_col
from .autoregression.v1 import AutoRegressionPredictor


class ARIMA:

    @classmethod
    def ma(cls: ARIMA, series: pd.DataFrame, col: Optional[str] = None, shift: int = 2):
        """Get the Moving Average value of the series

        Args:
            series (pd.DataFrame): DataFrame to handle
            col (Optional[str], optional): Name of the column, if more than one. Defaults to None.
            shift (int, optional): Number of shifts to compute the MA on. Defaults to 2.
        """
        df = series.copy()
        col = _get_col(series, col)
        shifted = df[col].rolling(shift+1).sum()[shift:]
        ma2_model = sm.tsa.arima.ARIMA(shifted, order=(0,0,shift)).fit()
        return pd.DataFrame({"MA" : shifted, "Predicted": ma2_model.predict()})


    @classmethod
    def predict_arma_coefficients(cls: ARIMA, series: pd.DataFrame, col: Optional[str] = None, ar_deg: int = 3, ma_deg: int = 2, steps: int = 5):
        """Perform an ARMA prediction using the lasat ar_deg values, and the last ma_deg errors. (AR and MA parts)

        Args:
            series (pd.DataFrame): DataFrame to handle
            col (Optional[str], optional): Column that contains the values. Defaults to None.
            ar_deg (int, optional): Ddegree of the AR part. Defaults to 3.
            ma_deg (int, optional): Degree of the MA part. Defaults to 2.
            steps (int, optional): Number of steps to iterate. Defaults to 5.
        """
        
        col = _get_col(series, col)
        ar_coeff = AutoRegressionPredictor.get_coefficients(series[col].values, ar_deg)
        ma_coeff = np.zeros(ma_deg)
        learning_rate = 0.0001
        for x in range(steps):
            ma_grad = np.zeros(ma_deg)
            last_errors = [0] * ma_deg
            predicted = [np.nan] * ar_deg
            for i in range(len(series[col]) - ar_deg):
                ar_pred = np.dot(ar_coeff, series[col][i:i+ar_deg])
                ma_pred = np.dot(ma_coeff, last_errors)
                pred = ar_pred + ma_pred
                predicted.append(pred)
                observed = series[col][i+ar_deg]
                error = observed - pred
                ma_pred_grad = np.array(last_errors)
                ar_pred_grad = np.array(series[col][i:i+ar_deg])

                last_errors = last_errors[1:] + [error]

                ma_grad = ma_grad - 2 * ma_pred_grad * (observed - pred)

            ma_coeff = ma_coeff - ma_grad * learning_rate / (len(series[col]) - ar_deg)

        return ar_coeff, ma_coeff

    @classmethod
    def predict(cls: ARIMA, series: pd.DataFrame, col: str = None, ar_deg: int = 3, ma_deg: int = 2, int_deg: int = 2, trend: str = 'n') -> pd.DataFrame:
        """Compute the ARIMA prediction given the parameters on the series[col]

        Args:
            series (pd.DataFrame): DataFrame that contains the information
            col (str, optional): Column name, if more than one are present in `series`. Defaults to None.
            ar_deg (int, optional): Degree for the AR part. Defaults to 3.
            ma_deg (int, optional): Degree for the MA part. Defaults to 2.
            int_deg (int, optional): Degree for the Integration part. Defaults to 2.
            trend (str, optional): Trend type for the ARIMA model. Defaults to 'n'.

        Returns:
            pd.DataFrame: The original dataframe with a new column containing the predicted values
        """
        series = series.copy()
        col = _get_col(series, col)
        model = sm.tsa.arima.ARIMA(series[col].values, order=(ar_deg, int_deg, ma_deg), trend=trend)
        res = model.fit().predict()
        series[f"Predicted {col}"] = res
        return series[1:]
