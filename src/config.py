from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
REPORT_DIR = BASE_DIR / "report"
REPORT_IMAGES_DIR = REPORT_DIR / "images"

RANDOM_STATE = 42
TARGET_COL = "Cover_Type"

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
ALL_FEATURES = NUM_COLS + WILDERNESS_COLS + SOIL_COLS