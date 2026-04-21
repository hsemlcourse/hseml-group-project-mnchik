import gzip
import shutil
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/Users/mnchk/hseml-group-project-mnchik").resolve()
ZIP_PATH = PROJECT_ROOT / "data/raw/covertype.zip"
INNER_GZ_PATH = "covtype.data.gz"

OUT_GZ_PATH = PROJECT_ROOT / "data/raw/covtype_bonus.data.gz"
OUT_DATA_PATH = PROJECT_ROOT / "data/raw/covtype_bonus.data"
OUT_CSV_PATH = PROJECT_ROOT / "data/raw/covertype_parsed_bonus.csv"

NUM_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]
TARGET_COL = "Cover_Type"
ALL_COLUMNS = NUM_COLS + WILDERNESS_COLS + SOIL_COLS + [TARGET_COL]

def extract_gz_from_zip() -> Path:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP не найден: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = zf.namelist()
        if INNER_GZ_PATH not in names:
            raise FileNotFoundError(
                f"В архиве нет {INNER_GZ_PATH}. Есть:\n" + "\n".join(names)
            )
        with zf.open(INNER_GZ_PATH) as f_in, open(OUT_GZ_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return OUT_GZ_PATH

def gunzip_to_data(gz_path: Path) -> Path:
    with gzip.open(gz_path, "rb") as f_in:
        with open(OUT_DATA_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return OUT_DATA_PATH

def parse_raw_data_to_csv(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, header=None, names=ALL_COLUMNS)
    df.to_csv(OUT_CSV_PATH, index=False)
    return df

def build_bonus_parsed_dataset() -> pd.DataFrame:
    gz_path = extract_gz_from_zip()
    data_path = gunzip_to_data(gz_path)
    df = parse_raw_data_to_csv(data_path)
    return df
