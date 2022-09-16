# TimeSeries

This project has several utlity functions and prediction methods to handle Time Series data. 
***
## Generate the Time Series

Time Series can be generated given a CSV, a Ticker (YFinance) or a Quandl code. The library transforms the data into a 
Pandas DataFrame with just one column in order to perform predictions, smoothen the values, plot them, etc. 

## Predict results

The submodule predictors contains several prediction method (ARIMA, AR, VAR, GARCH, Decompisition...) that are usfeul to predict the given data into the future. 
For example: 

```python
from timeseries.predictors import MonteCarloPredictor
v = MonteCarloPredictor.predict(btc_df, trajectory_length=100)

from timeseries.predictors.autoregression import AutoRegressionPredictorV1
v = AutoRegressionPredictorV1.predict(btc_df, order=4)
```

## Assess results

There is also an utility that perform an RMSE computation on the predicted resutlts so one knows how far from reality the values might be.

```python
from timeseries.assessor import PredictionAssessor

rmse = PredictionAssessor.evaluate_rmse(df["predicted"], df["actual"])
```

***
## DISCLAIMER
Since the repository is still under development, several bugs may be encountered if used. Feel free to comment advices/bugs! 🐞
