
import pandas as pd

from src.config import ALL_FEATURES, TARGET_COL


def validate_schema(df: pd.DataFrame) -> None:
    expected = set(ALL_FEATURES + [TARGET_COL])
    actual = set(df.columns)

    missing = expected - actual
    extra = actual - expected

    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")

def summarize_dataset(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "target_classes": sorted(df[TARGET_COL].unique().tolist()),
    }
