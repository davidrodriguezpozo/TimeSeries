import pandas as pd
import quandl as qd


class QuanDLDataFrame:
    """Class to wrap a QuanDL DataFrame that add some useful methods
    """
    def __init__(self, code: str, name:str):
        self.code = code
        self._df = qd.get(self.code)
        self.name = name

    def as_df(self) -> pd.DataFrame:
        return self._df

    def time_series(self) -> pd.DataFrame:
        return self._df.drop(columns=[col for col in self._df if col != 'Close'])
