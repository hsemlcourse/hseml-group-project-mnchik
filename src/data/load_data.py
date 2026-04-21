from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import RAW_DATA_DIR, TARGET_COL

from pathlib import Path
from typing import Union
import pandas as pd

def load_local_csv(path: Union[str, Path] = RAW_DATA_DIR / "covertype.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_from_ucirepo(save: bool = True) -> pd.DataFrame:
    dataset = fetch_ucirepo(id=31)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = X.copy()
    df[TARGET_COL] = y[TARGET_COL]

    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_DIR / "covertype.csv", index=False)

    return df

def load_raw_data() -> pd.DataFrame:
    csv_path = RAW_DATA_DIR / "covertype.csv"
    if csv_path.exists():
        return load_local_csv(csv_path)
    return load_from_ucirepo(save=True)
