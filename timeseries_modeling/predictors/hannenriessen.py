from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import _create_diff_col, _get_col
from .autoregression.v1 import AutoRegressionPredictor


class HannenRissanen:
    
    @classmethod
    def predict_arima_coefficients(cls: HannenRissanen, series: pd.DataFrame, col: str=None, ar_deg: int = 3, ma_deg: int = 2, steps: int = 50) -> np.ndarray:
        """Compute the Hannen-Rissanen prediction of the coefficients for an ARIMA model. This method only uses regression,
        no gradient descent at all. 

        Args:
            series (pd.DataFrame): DataFrame that contains the information
            col (str, optional): Name of the column (if more than one). Defaults to None.
            ar_deg (int, optional): Degree for the AR part. Defaults to 3.
            ma_deg (int, optional): Degree for the MA part. Defaults to 2.
            steps (int, optional): Number of steps to iterate. Defaults to 50.
        """
        col = _get_col(series, col)
        coeffs = np.append(AutoRegressionPredictor.get_coefficients(series[col].values, ar_deg), np.zeros(ma_deg, dtype=np.float64))
        errors = [0.0] * len(series[col])
        series = series[col].values
        for x in range(steps):
            design_matrix_rows = []
            for i in range(max(ar_deg, ma_deg), len(series)):
                values = np.append(series[i-ar_deg:i], errors[i-ma_deg:i])
                pred = np.dot(values, coeffs)
                design_matrix_rows.append(values)
                errors[i] = series[i] - pred
            design_matrix = np.array(design_matrix_rows, dtype=np.float64)
            coeffs = AutoRegressionPredictor.linear_regression(design_matrix, series[max(ar_deg, ma_deg):])
        return coeffs


    @classmethod
    def predict_diff(cls: HannenRissanen, series: pd.DataFrame, col: str=None, ar_deg: int = 3, ma_deg: int = 2, steps: int = 50):
        """Compute the Hannen-Rissanen prediction of the coefficients for an ARIMA model. This method only uses regression,
        no gradient descent at all. 

        Args:
            series (pd.DataFrame): DataFrame that contains the information
            col (str, optional): Name of the column (if more than one). Defaults to None.
            ar_deg (int, optional): Degree for the AR part. Defaults to 3.
            ma_deg (int, optional): Degree for the MA part. Defaults to 2.
            steps (int, optional): Number of steps to iterate. Defaults to 50.
        """
        col = _get_col(series, col)
        series = _create_diff_col(series, col, as_values=True)
        series = series[~np.isnan(series)]
        coeffs = np.append(AutoRegressionPredictor.get_coefficients(series, ar_deg), np.zeros(ma_deg, dtype=np.float64))
        errors = [0.0] * len(series)
        for x in range(steps):
            design_matrix_rows = []
            for i in range(max(ar_deg, ma_deg), len(series)):
                values = np.append(series[i-ar_deg:i], errors[i-ma_deg:i])
                pred = np.dot(values, coeffs)
                design_matrix_rows.append(values)
                errors[i] = series[i] - pred
            design_matrix = np.array(design_matrix_rows, dtype=np.float64)
            coeffs = AutoRegressionPredictor.linear_regression(design_matrix, series[max(ar_deg, ma_deg):])
        return coeffs
