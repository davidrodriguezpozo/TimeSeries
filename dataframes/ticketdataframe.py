import pandas as pd
import yfinance as yf


class TickerDataFrame:
    """Class to wrap a ticker DataFrame that adds extra methods
    """
    def __init__(self, ticker: str, name: str):
        self.ticker_code = ticker
        self.ticker = yf.Ticker(ticker)
        self.name = name

    def as_ticker(self) -> yf.Ticker:
        return self.ticker

    def as_df(self, start: str = None, end: str = None, period: str = None) -> pd.DataFrame:
        return self.ticker.history(start=start, end=end, period = period)

    def time_series(self, start: str, end: str, column: str) -> pd.DataFrame:
        return self.as_df(start=start, end=end).drop(columns=[col for col in self.as_df(period='1mo').columns if col != column])
