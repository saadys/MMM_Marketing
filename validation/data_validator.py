import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


class DataValidator:

    REQUIRED_COLUMNS = ["date", "sales"]

    def validate(self, df: pd.DataFrame) -> bool:
        for column in self.REQUIRED_COLUMNS:
            if column not in df.columns:
                raise ValueError(f"Missing column: {column}")

        if not is_datetime64_any_dtype(df["date"]):
            raise ValueError("Column 'date' must be a datetime type")

        if df["date"].isna().any():
            raise ValueError("Column 'date' contains missing values")

        if not is_numeric_dtype(df["sales"]):
            raise ValueError("Column 'sales' must be numeric")

        if df["sales"].lt(0).any():
            raise ValueError("Column 'sales' contains negative values")

        return True
    