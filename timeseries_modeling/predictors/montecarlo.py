from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import _get_col, rmse
from .autoregression.v2 import AutoRegressionPredictor


class MonteCarloPredictor:
    
    @classmethod
    def prob_between(cls, lower: float, higher: float, arr: np.ndarray) -> float:
        count = np.count_nonzero((arr > lower ) & (arr < higher))
        return count / arr.size

    @classmethod
    def predict(cls: MonteCarloPredictor, df: pd.DataFrame, col: str = None, trajectory_length: int =100) -> pd.DataFrame:
        """Use MonteCarlo algorithm to predict several future possiblities of the Time Series

        Args:
            df (pd.DataFrame): The DF that contains the values
            col (str, optional): Name of the column (if more than one). Defaults to None.
            trajectory_length (int, optional): Length of the predicted points. Defaults to 100.

        Returns:
            pd.DataFrame: The updated DF
        """
        col = _get_col(df, col)
        model = AutoRegressionPredictor.predict(df, lags=4, as_model=True)
        coeffs = np.fromiter(list(reversed(model.params)), dtype=float)
        num_coeffs = coeffs.shape[0]
        num_samples = 20
        initial_values = np.array(df[col][-10:].values).reshape(-1, 1)
        values = np.broadcast_to(initial_values, (initial_values.shape[0], num_samples))
        std_dev = rmse(model.predict(), df.reset_index()[col])
        for _ in range(trajectory_length): # The 100 sets how far in the future we are going
            next_prediction = coeffs @ values[-num_coeffs:,:]
            next_row = next_prediction + std_dev * np.random.randn(num_samples)
            values = np.append(values, next_row.reshape(1, num_samples), axis=0)
        return values


    @classmethod
    def get_probability(cls, values: pd.DataFrame, num_samples: int) -> float:
        threshold = .95 * values[0,9]
        booleans = values < threshold
        bad_rows= np.any(booleans,axis=1)
        bad_rows.sum()
        prob_stopping_out = bad_rows.sum()/num_samples
        return prob_stopping_out

    @classmethod
    def plot_probability_distribution(cls, values: np.ndarray) -> None:
        """Plot the probability density distribution of the series at the last period

        Args:
            values (np.ndarray): The values to compute the plot of
        """
        last_values = values[-1:][0]
        pd.DataFrame(last_values).plot.kde()
