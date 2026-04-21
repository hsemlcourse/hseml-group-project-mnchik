from pathlib import Path
from copy import deepcopy
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.config import RANDOM_STATE
from src.data.split_data import split_dataset
from src.models.metrics import compute_classification_metrics, format_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_CSV = EXPERIMENTS_DIR / "experiments.csv"


def load_features_data() -> pd.DataFrame:
    """Загрузить датасет с feature engineering для моделирования."""
    path = DATA_PROCESSED / "covertype_features.csv"
    return pd.read_csv(path)


def get_models() -> Dict[str, object]:
    """Собрать словарь моделей, участвующих в экспериментах."""
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "linear_svc": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LinearSVC(random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def run_all_models() -> pd.DataFrame:
    """
    Запустить все модели на одном и том же train/val/test‑сплите,
    посчитать метрики и записать результаты в experiments/experiments.csv.

    Возвращает DataFrame со всеми экспериментами.
    """
    df = load_features_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    models = get_models()
    experiments = []

    for name, model in models.items():
        print(f"- {name} -")
        clf = deepcopy(model)
        clf.fit(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        y_test_pred = clf.predict(X_test)

        val_metrics = format_metrics(
            compute_classification_metrics(y_val, y_val_pred)
        )
        test_metrics = format_metrics(
            compute_classification_metrics(y_test, y_test_pred)
        )

        row = {
            "model": name,
            "params": str(clf.get_params()),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_f1_weighted": test_metrics["f1_weighted"],
        }
        experiments.append(row)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    df_exp = pd.DataFrame(experiments)

    if EXPERIMENTS_CSV.exists():
        df_old = pd.read_csv(EXPERIMENTS_CSV)
        df_exp = pd.concat([df_old, df_exp], ignore_index=True)

    df_exp.to_csv(EXPERIMENTS_CSV, index=False)
    return df_exp


if __name__ == "__main__":
    df_exp = run_all_models()
    print(df_exp.sort_values("val_f1_macro", ascending=False).head())
