
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import quandl as qd
import statsmodels.api as sm
import yfinance as yf
from matplotlib import pyplot as plt

AutoCorrelationPredictor.plot_autocorrelation(apple_df, lags=40)
AutoCorrelationPredictor.plot_autocorrelation(coin_df, lags=40)
AutoCorrelationPredictor.plot_autocorrelation(temp_df, lags=40)
AutoCorrelationPredictor.plot_autocorrelation(amazon_df, lags=40)

apple_ar = AutoCorrelationPredictor.predict(apple_df, "value", 10)
coin_ar = AutoCorrelationPredictor.predict(coin_df, "value", 10)
temp_ar = AutoCorrelationPredictor.predict(temp_df, "value", 10)
amazn_ar = AutoCorrelationPredictor.predict(amazon_df, "value", 10)

results_table["Statsmodels AR"] = [PredictionAssessor.evaluate_rmse(df, "value") for df in [apple_ar, coin_ar, amazn_ar, temp_ar]]
results_table

new_df = AutoCorrelationPredictor.predict(suns_df.loc[suns_df.index > '1950'], "value")


AutoCorrelationPredictor.plot_autocorrelation(suns_df, lags=50)


btc = TickerDataFrame("BTC-USD", "Bitcoin USD")
btc_df = btc.time_series("2014-01-01", "2021-10-30", "Close")



v = MonteCarloPredictor.predict(btc_df, trajectory_length=100)
plt.plot(v)
plt.show()

MonteCarloPredictor.plot_probability_distribution(v)


         

smoothed = SmoothingLibrary.ma_smooth(apple_df, number=10)
smoothed = SmoothingLibrary.ma_smooth(smoothed, col="value", number=50)
smoothed = SmoothingLibrary.ma_smooth(smoothed, col="value", number=100)
smoothed = SmoothingLibrary.exponential_smooth(smoothed, col="value", alpha=0.5)
smoothed = SmoothingLibrary.exponential_smooth(smoothed, col="value", alpha=0.1)

# %%
smoothed.plot()


# %%
from __future__ import annotations

from typing import Union

from statsmodels.tsa.seasonal import STL

df = DecompositionLibrary.decompose_df(apple_df, alpha=0.2)
df = df.dropna()
df.plot()

# %%
new_df = DecompositionLibrary.predict(apple_df)
new_df.plot()

# %%
apple_lowess = DecompositionLibrary.predict_lowess(apple_df)
coin_lowess = DecompositionLibrary.predict_lowess(coin_df)
amazon_lowess = DecompositionLibrary.predict_lowess(amazon_df)
temp_lowess = DecompositionLibrary.predict_lowess(temp_df)

apple_stl = DecompositionLibrary.predict_STL(apple_df.resample('D').sum(), period=9) # We need a DatetimeIndex with freq so that ARIMA understands it)
coin_stl = DecompositionLibrary.predict_STL(coin_df.resample('D').sum(),period=9) # We need a DatetimeIndex with freq so that ARIMA understands it)
amazon_stl = DecompositionLibrary.predict_STL(amazon_df.resample('D').sum(), period=12) # We need a DatetimeIndex with freq so that ARIMA understands it
temp_stl = DecompositionLibrary.predict_STL(temp_df.resample('M').sum(), period=9) # We need a DatetimeIndex with freq so that ARIMA understands it

results_table["Lowess"] = [PredictionAssessor.evaluate_rmse(df, "value", "Lowess prediction value") for df in [apple_lowess, coin_lowess, amazon_lowess, temp_lowess]]
results_table["STL"] = [PredictionAssessor.evaluate_rmse(df, "value", "Prediction STL") for df in [apple_stl, coin_stl, amazon_stl, temp_stl]]
results_table

# %%
smoothed = DecompositionLibrary.predict(apple_lowess, col="value")
smoothed = DecompositionLibrary.predict_STL(smoothed, col="value", period=20)

# %%
smoothed.plot()

# %%
lowess_eval = PredictionAssessor.evaluate(smoothed, "value", "Lowess prediction value")
exp_eval = PredictionAssessor.evaluate(smoothed, "value", "Dec. prediction value")

# %%
print(lowess_eval)
print(exp_eval)

# %% [markdown]
# ## Fast Fourier Transform
# 
# Before getting into ARIMA models, we must have a glimpse of how to implement a Fast Fourier Transform in python. This can be done using `numpy` method `fft.fft()`. We can see an example below. 

# %%
def implement_fft(series: np.ndarray):
    return np.abs(np.fft.fft(series))

plt.plot(implement_fft(suns_df["value"].values)[10:len(suns_df)//5])

# %% [markdown]
# We can identify easily that there is a seasonality with period 18 + 10 = 28 for the suns datafarame

# %%
plt.plot(implement_fft(temp_df["value"].values)[10:len(temp_df)//5])

# %% [markdown]
# For the temperature DF, we can see a seasonality of 16 months pproximately. With these results, the period in the `predict_STL` method can be passed on with creatinity, instead of trying random numbers. 

# %%
new_df = DecompositionLibrary.predict_STL(suns_df, col="value", period=28)

# %%
new_df.plot()

    
new_df = apple_df.copy().resample('D').sum() # We need a DatetimeIndex with freq so that ARIMA understands it
predict = ARIMAPredictor.ma(new_df)
predict.plot()

# %% [markdown]
# Since the values for the shifted MA (the manual `.shift()`) are the sum of the previous `shift` values, the value of this column `MA` will always be greater than the predicted one. This is just an example of how the third parameter of the ARIMA model (x,x,shift) is the one that controls the moving average bit of the ARIMA model. 
# 
# Now we can predict the values for the ARMA model using `AutoRegression` and gradient descent to compute the coefficients for the AR bit. 

# %%
coefficients = ARIMAPredictor.ARMA_predict(new_df, "value")
print(coefficients)

# %% [markdown]
# Then we can do the same but using the Hannen Rissanen algorithm, which doesn't use gradient descent, but a series of Linear Regressions to compute the coefficients. 

# %%
coefficients_HR = ARIMAPredictor.HannenRissanen_predict(apple_df, "value")

# %%
coefficients_HR

# %% [markdown]
# Now, the `Augmented Dickey Fuller` algorithm will help us decide if a series has to be differentiated in order to obtain its coefficients. This has to do with the trending bit of the series. If a series has a trend, one way to eliminate this trend is to perform the differences between to timestamps and use these differences instead

# %%
from statsmodels.tsa.stattools import adfuller


def is_stationary(series: np.ndarray):
    return adfuller(series)[1] < 0.05

is_stationary(apple_df)

# %% [markdown]
# We get that the apple_df is not stationary (it can be seen in the previous cells that it has a trend upwards). Then we can use the same method to test if the first difference is stationary. 

# %%
diff_df = TimeSeriesLibrary._create_diff_col(apple_df, "value", as_values=True)
is_stationary(diff_df[~np.isnan(diff_df)])

# %% [markdown]
# We can see how after a single difference between two consecutive timestmaps, the time series has become stationary. We can then use the adapted Hannen Rissanen method that uses the differences column instead. 

# %%
coefficients_3 = ARIMAPredictor.HannenRissanen_predict_differentiate(apple_df, "value")
coefficients_3

# %% [markdown]
# then the only thing left to do is to use the ARIMA `predict` model to obtain the predicted values of the series, we'll do this for all the previous dataframes and compare the results with the other models.

# %%
apple_arima = ARIMAPredictor.predict(apple_df, "value", 2, 10, 1)
coin_arima = ARIMAPredictor.predict(coin_df, "value", 2, 10, 1)
amazon_arima = ARIMAPredictor.predict(amazn_df, "value", 3, 10, 1)
temp_arima = ARIMAPredictor.predict(temp_df, "value", 2, 10, 1)

# %%
results_table["ARIMA"] = [PredictionAssessor.evaluate_rmse(df, "value", "Predicted value") for df in [apple_arima, coin_arima, amazon_arima, temp_arima]]
results_table

# %% [markdown]
# We can see how the ARIMA model outperforms all the other models in almost all the cases. This means that the ARIMA model is actually a really good approximation and prediction algorithm. However, these differences are very small and it is very hard to distinguis among them in a single plot. We can, however, plot the ARIM predictions for the `Apple` dataframe. 

# %%
apple_arima.plot()

# %%
v

# %% [markdown]
# ***
# # ARCH and Variance
# ***

# %%
from __future__ import annotations

from abc import abstractclassmethod


class ARCHModel(TimeSeriesLibrary):
    
    @classmethod
    def local_variance(cls: ARCHModel, series: np.ndarray) -> np.ndarray:
        var = np.zeros(len(series))
        for i in range(5, len(series)- 5):
            new_var = np.var(series[i-5:i+5])
            var[i] = new_var
        
        for i in range(len(series) - 5, len(series)):
            var[i] = new_var
        return var
    
    @classmethod
    def compute_predicted_error_df(cls: ARCHModel, series: pd.DataFrame, col: str = None, ar_deg: int = 4, arch_deg: int = 4) -> pd.DataFrame:
        series = series.copy()
        col = cls._get_col(series, col)
        predicted = sm.tsa.AutoReg(series[col].values, lags=ar_deg, trend='n').fit().predict()
        series["Predicted"] = predicted
        error_sq = (series["Predicted"] - series[col]) ** 2
        error_sq[error_sq.isna()] = 0
        error_predicted = sm.tsa.AutoReg(error_sq.values, lags=arch_deg, trend='c').fit().predict()
        series["Error Predicted"] = error_predicted
        series["Error"] = error_sq
        return series

    @classmethod
    def get_lower_upper_std(cls: ARCHModel, series: pd.DataFrame, col: str = None, ar_deg: int = 4, arch_deg: int = 4) -> pd.DataFrame:
        col = cls._get_col(series, col)
        series = cls.compute_predicted_error_df(series, col, ar_deg, arch_deg)
        error_prediction = np.sqrt(series["Error Predicted"].values)
        series["Lower"] = series[col] - error_prediction
        series["Upper"] = series[col] + error_prediction
        return series

    @classmethod
    def get_percentages(cls: ARCHModel, series: pd.DataFrame, col: str = None, ar_deg: int = 4, arch_deg: int = 4) -> Tuple[float, float]:
        col = cls._get_col(series, col)
        series = cls.get_lower_upper_std(series, col, ar_deg, arch_deg)
        error_prediction = np.sqrt(series["Error Predicted"].values)
        in_one_sd = (series['Upper'] > series[col]) & (series["Lower"] < series[col])
        in_two_sd = (series['Predicted'] + 2 * error_prediction > series[col]) & (series['Predicted'] - 2 * error_prediction < series[col])
        return in_one_sd.mean(), in_two_sd.mean()

class GARCHModel(ARCHModel):

    @classmethod
    def compute_predicted_error_df(cls: ARCHModel, series: pd.DataFrame, col: str = None, ar_deg: int = 4, arch_deg: int = 4) -> pd.DataFrame:
        series = series.copy()
        col = cls._get_col(series, col)
        predicted = sm.tsa.AutoReg(series[col].values, lags=ar_deg, trend='n').fit().predict()
        series["Predicted"] = predicted
        error_sq = (series["Predicted"] - series[col]) ** 2
        error_sq[error_sq.isna()] = 0
        error_predicted = sm.tsa.arima.ARIMA(error_sq.values, order=(arch_deg, 0, 3), trend='c').fit().predict()
        series["Error Predicted"] = error_predicted
        series["Error"] = error_sq
        return series

df_arch = ARCHModel.get_lower_upper_std(apple_df)
df_arch.drop(columns=["Error", "Error Predicted", "value"]).plot()
df_garch = GARCHModel.get_lower_upper_std(apple_df)
df_garch.drop(columns=["Error", "Error Predicted", "value"]).plot()
print(ARCHModel.get_percentages(apple_df))
print(GARCHModel.get_percentages(apple_df))

# %%
PredictionAssessor.rmse(df_arch["Predicted"], df_arch["value"])

# %%
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

# %%
eth_code = "ETH-USD"
eth_ticker = TickerDataFrame(eth_code, "Ethereum")
eth_df = eth_ticker.time_series("2014-01-01", "2021-10-30", "Close")

combined_df = pd.DataFrame()
combined_df["ethereum"] = eth_df["Close"]
combined_df["bitcoin"] = btc_df["Close"]
predicted = VARModel.predict(combined_df, 4)

# %%
predicted

# %%
combined_df

# %%
evaluation = PredictionAssessor.rmse(predicted["ethereum"], combined_df["ethereum"])

# %%
plot_df = pd.DataFrame()
plot_df["Actual"] = combined_df["ethereum"].values
plot_df["Predicted"] = predicted["ethereum"].values

# %%
plot_df.plot()

# %%


# %%
combined_df

# %%
def covariance(x:np.ndarray, y: np.ndarray) -> float:
    return (x*y).mean() - x.mean() * y.mean()

def beta(x: np.ndarray, y:np.ndarray) -> float:
    return covariance(x, y) / covariance(x, x)

def root_square_matrix(cov: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(cov)

# %%
from typing import Any


class KalmanFilter:
    def __init__(self, F, Q, R, H, obs):
        self.F: np.ndarray = F
        self.Q: np.ndarray = Q
        self.R: float = R
        self.H: np.ndarray = H
        self.observations: List[List[Any]] = obs

    def kalman_step(self, state_covariance: np.ndarray, state_estimate: np.ndarray, i: int) -> Tuple[np.ndarray, np.ndarray]:
        observation = self.observations[i]
        predicted_state = self.F @ state_estimate
        predicted_covariance = self.F @ state_covariance @ self.F.T + self.Q
        prefit_residual = observation - self.H @ predicted_state
        prefit_covariance = self.H @ predicted_covariance @ self.H.T + self.R
        kalman_gain = (predicted_covariance @ np.transpose(self.H)) @ np.linalg.inv(prefit_covariance)
        state_estimate = predicted_state + kalman_gain @ prefit_residual
        state_covariance = (np.identity(len(state_estimate)) - kalman_gain @ self.H) @ state_covariance
        return state_estimate, state_covariance

    def kalman_multi(self, state_covariance: np.ndarray, state_estimate: np.ndarray) -> None:
        predictions = []
        for i in range(len(self.observations)):
            state_estimate, state_covariance = self.kalman_step(state_covariance, state_estimate, i)
            predictions.append(self.H @ state_estimate)
        return predictions

# %%
F = np.matrix('1, 2; 0, 1')
Q = np.matrix('1, 1; 1,1')
R = 2
P = np.matrix('1 0; 0 1')
H = np.matrix('1, 0')
observations = [[125]]

kf1 = KalmanFilter(F, Q, R, H, observations)
state_0 = np.array([100, 10])
state = state_0.reshape(-1, 1)
s, c = kf1.kalman_step(P, state, 0)

# %%
observations = [[125], [142],[163], [184], [200]]
kf1 = KalmanFilter(F, Q, R, H, observations)
state_0 = np.array([100, 10])
state = state_0.reshape(-1, 1)
kf1.kalman_multi(P, state)

# %%
df = pd.DataFrame({"col": [1,2,3.5,2,6, 5.6]})
df100 = 100*df
pred = AutoRegressionPredictor.predict(df, order=2)

# %%
pred = AutoRegressionPredictor.get_coefficients(df["col"].values, order=2)

# %%
pred100 = AutoRegressionPredictor.get_coefficients(df100["col"].values, order=2)

# %%
pred100

# %%
pred

# %%
values = [5,6,5,7,9,10,9,12,11,12,9,8]
values_march_dec = values[2:]
df = pd.DataFrame({"col":values_march_dec})

pred_df = PredictionTimeSeries.average_trend_prediction(df, last=2)
pred_df

# %%



