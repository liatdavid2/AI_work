import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, f1_score

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ======================================================
# Paths & constants
# ======================================================

DATA_PATH = "training/data/merged/training_candidates.parquet"
MODEL_DIR = "training/models"

LABEL_COL = "Label"

TARGET_CLASSES = [
    "generic",
    "normal",
    "reconnaissance",
    "shellcode",
    "worms",
]

N_FEATURES = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ======================================================
# Training
# ======================================================

def train_xgb():
    if not os.path.exists(DATA_PATH):
        print("Merged dataset not found. Skipping training.")
        return

    df = pd.read_parquet(DATA_PATH)

    if LABEL_COL not in df.columns:
        print("Label column not found. Skipping training.")
        return

    # keep only valid target labels
    df = df[df[LABEL_COL].isin(range(len(TARGET_CLASSES)))].copy()
    if df.empty:
        print("No samples for target classes. Skipping training.")
        return

    # select numeric features
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != LABEL_COL]

    X_all = df[numeric_cols]
    y_all = df[LABEL_COL]

    # remove constant columns
    X_all = X_all.loc[:, X_all.nunique() > 1]

    # balance classes (downsample to min class size)
    df_all = pd.concat([X_all, y_all], axis=1)
    min_class_size = df_all[LABEL_COL].value_counts().min()

    balanced_df = pd.concat(
        [
            resample(
                df_all[df_all[LABEL_COL] == cls],
                replace=False,
                n_samples=min_class_size,
                random_state=RANDOM_STATE,
            )
            for cls in df_all[LABEL_COL].unique()
        ]
    ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X = balanced_df.drop(columns=LABEL_COL)
    y = balanced_df[LABEL_COL]

    # fill missing values
    X = X.astype(float).fillna(X.median(numeric_only=True))

    # scale + feature selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=N_FEATURES)
    with np.errstate(divide="ignore", invalid="ignore"):
        X_selected = selector.fit_transform(X_scaled, y)

    selected_columns = X.columns[selector.get_support()].tolist()

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(TARGET_CLASSES),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # evaluation
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("=== XGBoost – 5-Class Classification ===")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(TARGET_CLASSES)),
            target_names=TARGET_CLASSES,
        )
    )
    print(f"Weighted F1: {f1:.4f}")

    # save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)

    date_str = datetime.utcnow().strftime("%Y%m%d")
    model_filename = f"xgb_5labels_f1_{f1:.4f}_{date_str}.joblib"
    model_path = os.path.join(MODEL_DIR, model_filename)

    features_path = os.path.join(MODEL_DIR, "features.joblib")

    joblib.dump(model, model_path)
    joblib.dump(
        {
            "selected_columns": selected_columns,
            "scaler": scaler,
            "selector": selector,
            "target_classes": TARGET_CLASSES,
            "label_col": LABEL_COL,
            "n_features": N_FEATURES,
        },
        features_path,
    )

    print(f"Model saved to: {model_path}")
    print(f"Feature artifacts saved to: {features_path}")


if __name__ == "__main__":
    train_xgb()
