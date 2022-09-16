import pandas as pd


class CSVDataFrame:
    def __init__(self, path: str, name: str, index: str = "Date"):
        self.path = path
        self._df = pd.read_csv(path).set_index(index)
        self.name = name

    def as_df(self) -> pd.DataFrame:
        return self._df

    def time_series(self, column: str = None) -> pd.DataFrame:
        column = column or "Close"
        df = self._df.drop(columns=[col for col in self._df.columns if col != column])
        df[column] = df[column].replace(",", "", regex=True).astype('float')
        return df
