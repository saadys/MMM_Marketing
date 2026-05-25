from typing import List

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


class SilverTransformer:

    WEEKLY_FREQUENCY = "W-MON"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" not in df.columns:
            raise ValueError("Column 'date' is required for transformation")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("Cannot transform dataset because some dates are invalid")

        df = df.sort_values("date").set_index("date")

        if "sales" not in df.columns:
            raise ValueError("Column 'sales' is required for transformation")

        weekly_sales = self._resample_sales(df["sales"])

        raw_channels = df.drop(columns=["sales"])
        numeric_channels = raw_channels.select_dtypes(include="number")
        channel_columns = [
            col
            for col in numeric_channels.columns
            if numeric_channels[col].nunique(dropna=True) / max(len(numeric_channels), 1) < 0.95
        ]
        budget_columns = numeric_channels[channel_columns]
        weekly_budget = self._resample_channels(budget_columns)

        weekly = pd.concat([weekly_sales, weekly_budget], axis=1)
        weekly = weekly.sort_index()
        weekly = self._ensure_weekly_index(weekly)
        weekly = weekly.fillna(0).reset_index()

        return weekly

    def _resample_sales(self, sales: pd.Series) -> pd.Series:
        return sales.resample(self.WEEKLY_FREQUENCY).sum(min_count=1)

    def _resample_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.resample(self.WEEKLY_FREQUENCY).sum(min_count=1)

        weekly_data: List[pd.Series] = []

        for column in df.columns:
            series = df[column].astype(float)
            frequency = pd.infer_freq(series.dropna().index)

            if frequency is None or frequency.startswith("D") or frequency.startswith("W"):
                weekly = series.resample(self.WEEKLY_FREQUENCY).sum(min_count=1)
            else:
                weekly = (
                    series.resample(self.WEEKLY_FREQUENCY)
                    .last()
                    .ffill()
                    .fillna(0)
                )

            weekly_data.append(weekly.rename(column))

        return pd.concat(weekly_data, axis=1)

    def _ensure_weekly_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if not is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        return df.asfreq(self.WEEKLY_FREQUENCY, method="ffill")
