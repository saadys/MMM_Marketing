from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.csv_loader import CSVLoader

from validation.data_validator import (
    DataValidator
)

from validation.anomaly_detector import (
    DetectAnomalie
)

from transformation.silver_transformer import (
    SilverTransformer
)

from gold.gold_builder import GoldBuilder


def main() -> None:
    loader = CSVLoader()

    df = loader.load_csv(
        "data/raw/ConsumerElectronics (1).csv"
    )

    validator = DataValidator()
    validator.validate(df)

    detector = DetectAnomalie()
    detector.detect(df)

    transformer = SilverTransformer()
    silver_df = transformer.transform(df)

    silver_df.to_csv(
        "data/silver/silver_sales.csv"
    )

    builder = GoldBuilder()
    gold_dataset = builder.build(silver_df)

    print("Pipeline exécuté avec succès")


if __name__ == "__main__":
    main()
