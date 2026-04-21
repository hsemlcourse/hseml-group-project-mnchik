import pandas as pd
from src.config import TARGET_COL, NUM_COLS, WILDERNESS_COLS, SOIL_COLS

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in WILDERNESS_COLS + SOIL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")

    return df

def report_data_quality(df: pd.DataFrame) -> dict:
    return {
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "numeric_summary": df.describe().to_dict(),
    }

def check_binary_integrity(df: pd.DataFrame) -> dict:
    result = {}

    for col in WILDERNESS_COLS + SOIL_COLS:
        values = sorted(df[col].dropna().unique().tolist())
        result[col] = values

    return result
