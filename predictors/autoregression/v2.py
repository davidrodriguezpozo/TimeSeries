import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels import api as sm

from ...utils import _get_col


class AutoRegressionPredictor:

    @classmethod
    def plot_autocorrelation(cls, df: pd.DataFrame, col: str = None, lags: int = 20) -> None:
        col = _get_col(df, col)
        values = df.reset_index()[col]
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(values, lags=lags, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(values, lags=lags, ax=ax2)

    @classmethod
    def predict(cls, df: pd.DataFrame, col: str = None, lags: int = 20, as_model: bool = False, trend: str = 'c') -> pd.DataFrame:
        df = df.copy()
        col = _get_col(df, col)
        values = np.log(df.reset_index()[col])
        results = sm.tsa.AutoReg(values, lags=lags, trend=trend).fit()
        if as_model:
            return results
        predicted_logs = results.predict()
        predicted_logs.index = df.index
        df[f"Prediction_{col}"] = np.exp(predicted_logs)
        return df
