class DetectAnomalie:

    def detect(self, df):
        if "sales" not in df.columns:
            raise ValueError("Column 'sales' is required for anomaly detection")

        missing_sales = int(df["sales"].isna().sum())
        missing_ratio = missing_sales / max(len(df), 1)

        if missing_sales > 100 and missing_ratio > 0.01:
            raise ValueError(
                f"Too many missing sales values: {missing_sales} rows ({missing_ratio:.2%})"
            )

        if (df["sales"] < 0).any():
            negative_sales = int((df["sales"] < 0).sum())
            raise ValueError(
                f"Negative sales values detected: {negative_sales} rows"
            )

        return True