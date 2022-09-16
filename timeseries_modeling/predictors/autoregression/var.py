import math

import numpy as np
import pandas as pd

from .v1 import AutoRegressionPredictor


class VARModel(AutoRegressionPredictor):

    @classmethod
    def train_var(cls, series: pd.DataFrame, order: int) -> np.ndarray:
        params_rows = []
        for col in series.columns:
            target_vector = np.array(series[col][order:])
            lagged_values = []
            for i in range(len(series[col]) - order):
                design_row = np.zeros(0)
                for param_column in series.columns:
                    design_row = np.append(design_row, series[param_column][i:i+order])
                lagged_values.append(design_row)
            design_matrix = np.array(lagged_values)
            params_rows.append(cls.linear_regression(design_matrix, target_vector))
        return np.array(params_rows)

    @classmethod
    def predict(cls, series: pd.DataFrame, order: int = 4) -> pd.DataFrame:
        params = cls.train_var(series, order)
        results = pd.DataFrame()
        for column_num, col in enumerate(series.columns):
            predicted_values = [math.nan] * order
            for i in range(len(series[col]) - order):
                lags = np.zeros(0)
                for param_column in series.columns:
                    lags = np.append(lags, series[param_column][i:i+order])
                predicted_values.append(np.dot(lags, params[column_num,:]))
            results[col] = predicted_values
        return results
