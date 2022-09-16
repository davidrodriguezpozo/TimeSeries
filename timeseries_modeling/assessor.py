from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .utils import _get_col, rmse


class PredictionAssessor:
    @classmethod
    def evaluate(cls: PredictionAssessor, df: pd.DataFrame, col: Optional[str] = None, pred_col: Optional[str] = None) -> float:
        col = _get_col(df, col)
        pred_col = pred_col or "Prediction_" + col
        error_sqr = 0
        total_pred = 0
        values = df[col]
        predicted = df[pred_col]
        for value, pred in zip(values, predicted):
            if pred > 0 and value > 0:
                error_sqr += (pred-value)**2
                total_pred += 1
        return np.sqrt(error_sqr/total_pred)

    @classmethod
    def evaluate_rmse(cls: PredictionAssessor, df: pd.DataFrame, col: Optional[str] = None, pred_col: Optional[str] = None) -> float:
        col = _get_col(df, col)
        pred_col = pred_col or "Prediction_" + col
        values = df[col]
        predicted = df[pred_col]
        return rmse(predicted, values)
