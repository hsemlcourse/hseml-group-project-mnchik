from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Вычисляет базовый набор метрик для многоклассовой классификации.
    Основная метрика для сравнения моделей — f1_macro.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def format_metrics(metrics: Dict[str, float], digits: int = 4) -> Dict[str, float]:
    """Округление метрик для красивого вывода и записи в таблицу."""
    return {k: float(np.round(v, digits)) for k, v in metrics.items()}
