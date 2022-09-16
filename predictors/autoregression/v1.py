

from typing import Optional

import numpy as np
import pandas as pd

from ...utils import _get_col


class AutoRegressionPredictor:
    @classmethod
    def linear_regression(cls, design_matrix: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
            """Performs the linear regression of the given arrays

            Args:
                design_matrix (np.ndarray): First operand (design matrix)
                target_vector (np.ndarray): Second operand (target vector)

            Returns:
                np.ndarray: The computed array
            """
            return np.linalg.inv(design_matrix.transpose() @ design_matrix) @ design_matrix.transpose() @ target_vector
    
    @classmethod
    def get_coefficients(cls, values: np.ndarray, order: int) -> np.ndarray:
        target_vector = np.array(values[order:])
        lagged_values = list()

        for i in range(len(values) - order):
            lagged_values.append(values[i:i+order])
        design_matrix = np.array(lagged_values)
        
        return cls.linear_regression(design_matrix, target_vector)
    
    @classmethod
    def predict_ar(cls, values: np.ndarray, order: int) -> np.ndarray:
        coefficients = cls.get_coefficients(values, order)
        predicted_values = [np.nan] * len(coefficients)
        for i in range(len(coefficients), len(values)):
            predicted_values.append(np.dot(coefficients, values[i-len(coefficients):i]))
        return np.array(predicted_values)
    
    @classmethod
    def predict(cls, df: pd.DataFrame, order: int, col: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        col = _get_col(df, col)
        values = df[col].dropna()
        predicted_values = cls.predict_ar(values, order)
        df[f"Prediction_{col}"] = predicted_values
        return df
