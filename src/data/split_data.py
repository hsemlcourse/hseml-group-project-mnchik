from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, RANDOM_STATE

def split_dataset(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test