from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class MMMDataset:
    dates: pd.DatetimeIndex
    features: pd.DataFrame
    target: pd.Series
    channel_names: List[str]
    metadata: Dict[str, Optional[object]]


class GoldBuilder:

    def build(self, df: pd.DataFrame) -> MMMDataset:
        if "sales" not in df.columns:
            raise ValueError("Column 'sales' is required to build the gold dataset")

        working = df.copy()
        if "date" in working.columns:
            working["date"] = pd.to_datetime(working["date"], errors="coerce")
            if working["date"].isna().any():
                raise ValueError("Gold dataset cannot contain invalid dates")
            working = working.set_index("date")
        elif working.index.name != "date":
            raise ValueError("Gold dataset requires a 'date' index or a 'date' column")

        features = working.drop(columns=["sales"])
        target = working["sales"]

        return MMMDataset(
            dates=working.index,
            features=features,
            target=target,
            channel_names=list(features.columns),
            metadata={
                "rows": len(working),
                "columns": len(working.columns),
                "feature_columns": list(features.columns),
            },
        )
