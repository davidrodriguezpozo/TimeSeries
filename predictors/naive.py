from __future__ import annotations

from typing import Optional

import pandas as pd

from ..utils import _create_diff_col, _get_col


class NaivePredictor:
    @classmethod
    def naive_prediction(cls: NaivePredictor, df: pd.DataFrame, col: Optional[str] = None) -> pd.DataFrame:
        """Performs a naive prediction, which is predicting that the current value is the same as the previous value. 

        Args:
            df (pd.DataFrame): DataFrame that contains the information
            col (Optional[str], optional): NAme of the column, if more than one. Defaults to None.

        Returns:
            pd.DataFrame: The modified dataframe
        """
        df = df.copy()
        col = _get_col(df, col)
        df[f"Prediction_{col}"] = df[col].shift()
        return df.dropna(axis=0)

    @classmethod
    def average_prediction(cls: NaivePredictor, df: pd.DataFrame, col: Optional[str] = None, last: Optional[int] = 2) -> pd.DataFrame:
        """Performs an average, perdiction, where the current value is the mean of the previos `last` values

        Args:
            df (pd.DataFrame): DataFrame that contains the information
            col (Optional[str], optional): Name of the column. Defaults to None.
            last (Optional[int], optional): Number of values to compute the average of. Defaults to 2.

        Returns:
            pd.DataFrame: The original dataframe, with an added column containing the predicted values
        """
        df = df.copy()
        col = _get_col(df, col)
        df[f"Prediction_{col}"] = 1/last * (df[col].cumsum().shift() - df[col].cumsum().shift(last+1))
        return df.dropna(axis=0)
    
    @classmethod
    def average_trend_prediction(cls: NaivePredictor, df: pd.DataFrame, col: Optional[str] = None, last: Optional[int] = 2) -> pd.DataFrame:
        """Computes the prediction using averages and differences, using the formula `Current value = Previous + Predicted diff`.
        Predicted diffrences are computed using an average prediction

        Args:
            df (pd.DataFrame): Dataframe that contains the column
            col (Optional[str], optional): Column name. Defaults to None.
            last (Optional[int], optional): Used for the average_prediction of the differences. Defaults to 2.

        Returns:
            pd.DataFrame: Originsal dataframe with the added column
        """
        df = df.copy()
        col = _get_col(df, col)
        df = _create_diff_col(df, col)
        df = cls.average_prediction(df, "diff", last)
        df[f"Prediction_{col}"] = df[col].shift() + df[f"Prediction_diff"]
        df = df.drop(columns=["diff", "Prediction_diff"]).dropna(axis=0)
        return df
