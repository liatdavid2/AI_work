import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "training/data/merged/training_candidates.parquet"
OUT_DIR = "training/data/splits"

MIN_LABELED_SAMPLES = 50
LABEL_COL = "Label"


def split():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)

    if LABEL_COL not in df.columns:
        print("No Label column found. Skipping split.")
        return

    labeled = df[df[LABEL_COL].notna()].copy()

    if len(labeled) < MIN_LABELED_SAMPLES:
        print(
            f"Not enough labeled samples for training "
            f"({len(labeled)} found, {MIN_LABELED_SAMPLES} required). "
            "Skipping split."
        )
        return

    train, temp = train_test_split(
        labeled,
        test_size=0.3,
        random_state=42,
        stratify=labeled[LABEL_COL],
    )

    valid, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42,
        stratify=temp[LABEL_COL],
    )

    train.to_parquet(f"{OUT_DIR}/train.parquet", index=False)
    valid.to_parquet(f"{OUT_DIR}/valid.parquet", index=False)
    test.to_parquet(f"{OUT_DIR}/test.parquet", index=False)

    print(
        f"Split completed: "
        f"train={len(train)}, valid={len(valid)}, test={len(test)}"
    )


if __name__ == "__main__":
    split()
