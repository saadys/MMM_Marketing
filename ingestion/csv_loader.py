from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel


class CSVLoaderSchema(BaseModel):
    required_columns: List[str] = ["date", "sales"]
    date_column: str = "date"
    numeric_columns: Optional[List[str]] = None


class CSVLoader:

    DATE_ALIASES = {
        "date",
        "order_date",
        "orderdate",
        "timestamp",
        "created_at",
        "date_time",
    }
    SALES_ALIASES = {
        "sales",
        "gmv",
        "revenue",
        "amount",
        "total",
        "sales_value",
    }

    def __init__(self, schema: Optional[CSVLoaderSchema] = None):
        self.schema = schema or CSVLoaderSchema()

    def load_csv(self, path: str) -> pd.DataFrame:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Source file not found: {csv_path}")

        df = pd.read_csv(csv_path, low_memory=False)
        df = self._normalize_column_names(df)
        df = self._normalize_date(df)
        self._validate_columns(df)
        df = self._coerce_numeric_columns(df)
        return df

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = {}
        lowercase_columns = {col.lower(): col for col in df.columns}

        for alias in self.DATE_ALIASES:
            if alias in lowercase_columns:
                normalized[lowercase_columns[alias]] = self.schema.date_column
                break

        for alias in self.SALES_ALIASES:
            if alias in lowercase_columns:
                normalized[lowercase_columns[alias]] = self.schema.required_columns[1]
                break

        if normalized:
            df = df.rename(columns=normalized)

        return df

    def _normalize_date(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace({"\\N": pd.NA, "NA": pd.NA, "na": pd.NA, "": pd.NA})
        df = df.replace(r'^\s*$', pd.NA, regex=True)

        if self.schema.date_column not in df.columns:
            df[self.schema.date_column] = pd.date_range(
                start="2000-01-01", periods=len(df), freq="D"
            )

        df[self.schema.date_column] = pd.to_datetime(
            df[self.schema.date_column], errors="coerce"
        )

        if df[self.schema.date_column].isna().any() and self.schema.date_column in df.columns:
            invalid_count = int(df[self.schema.date_column].isna().sum())
            raise ValueError(
                f"{invalid_count} invalid values in column '{self.schema.date_column}'"
            )

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        for required_column in self.schema.required_columns:
            if required_column not in df.columns:
                raise ValueError(f"Missing required column: {required_column}")

    def _coerce_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = self.schema.numeric_columns
        if numeric_columns is None:
            numeric_columns = [self.schema.required_columns[1]]

        for numeric_column in numeric_columns:
            if numeric_column not in df.columns:
                continue

            original = df[numeric_column]
            cleaned = original.astype("string").str.strip()
            cleaned = cleaned.replace({"": pd.NA, "NA": pd.NA, "na": pd.NA})
            coerced = pd.to_numeric(cleaned, errors="coerce")
            invalid_mask = cleaned.notna() & coerced.isna()
            if invalid_mask.any():
                invalid_values = cleaned[invalid_mask].unique().tolist()
                raise ValueError(
                    f"Column '{numeric_column}' contains invalid values: {invalid_values[:10]}"
                )

            df[numeric_column] = coerced

        return df