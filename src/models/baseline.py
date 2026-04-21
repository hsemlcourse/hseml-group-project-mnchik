from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.config import TARGET_COL, RANDOM_STATE
from src.data.split_data import split_dataset
from src.models.metrics import compute_classification_metrics, format_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_CSV = EXPERIMENTS_DIR / "experiments.csv"


def load_clean_data() -> pd.DataFrame:
    path = DATA_INTERIM / "covertype_clean.csv"
    return pd.read_csv(path)


def build_baseline_pipeline() -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]
    )
    return pipe


def run_baseline() -> Dict[str, Dict[str, float]]:
    df = load_clean_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    pipe = build_baseline_pipeline()
    pipe.fit(X_train, y_train)

    y_val_pred = pipe.predict(X_val)
    y_test_pred = pipe.predict(X_test)

    val_metrics = format_metrics(
        compute_classification_metrics(y_val, y_val_pred)
    )
    test_metrics = format_metrics(
        compute_classification_metrics(y_test, y_test_pred)
    )

    # записываем эксперимент
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "model": "logreg_baseline",
        "params": str(pipe.get_params()),
        "val_accuracy": val_metrics["accuracy"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_f1_weighted": val_metrics["f1_weighted"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_weighted": test_metrics["f1_weighted"],
    }

    if EXPERIMENTS_CSV.exists():
        df_exp = pd.read_csv(EXPERIMENTS_CSV)
        df_exp = pd.concat([df_exp, pd.DataFrame([row])], ignore_index=True)
    else:
        df_exp = pd.DataFrame([row])
    df_exp.to_csv(EXPERIMENTS_CSV, index=False)

    return {"val": val_metrics, "test": test_metrics}


if __name__ == "__main__":
    metrics = run_baseline()
    print("Baseline validation metrics:", metrics["val"])
    print("Baseline test metrics:", metrics["test"])