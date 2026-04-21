import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TARGET_COL, RANDOM_STATE
from src.data.split_data import split_dataset
from src.models.metrics import compute_classification_metrics
...
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import TARGET_COL, RANDOM_STATE
from src.data.split_data import split_dataset
from src.models.metrics import compute_classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def test_smoke_full_pipeline():
    # 1. Загружаем небольшой сэмпл данных
    df = pd.read_csv(DATA_PROCESSED / "covertype_features.csv")
    df_sample = df.sample(n=1000, random_state=RANDOM_STATE)

    # 2. Сплит (тот же, что в проекте)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df_sample)

    # 3. Простая модель (RandomForest)
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)

    # 4. Метрики и sanity-check
    metrics = compute_classification_metrics(y_val, y_val_pred)

    assert metrics["accuracy"] > 0.1
    assert metrics["f1_macro"] > 0.1