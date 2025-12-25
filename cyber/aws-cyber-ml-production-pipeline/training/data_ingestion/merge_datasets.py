import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASELINE_PATH = "training/data/baseline/UNSW_baseline.parquet"
LABELED_DIR = "training/data/labeled_from_prod"
OUTPUT_PATH = "training/data/merged/training_candidates.parquet"

# Target classes (fixed order)
TARGET_CLASSES = [
    "generic",
    "normal",
    "reconnaissance",
    "shellcode",
    "worms",
]


def load_labeled_from_prod():
    if not os.path.exists(LABELED_DIR):
        return []

    dfs = []
    for f in sorted(os.listdir(LABELED_DIR)):
        if f.endswith(".parquet"):
            dfs.append(pd.read_parquet(os.path.join(LABELED_DIR, f)))
    return dfs


def merge():
    os.makedirs("training/data/merged", exist_ok=True)

    dfs = []

    # baseline (always)
    baseline = pd.read_parquet(BASELINE_PATH)
    baseline["source"] = "baseline"
    dfs.append(baseline)

    # labeled batches from prod (optional)
    for df in load_labeled_from_prod():
        df["source"] = "labeled_from_prod"
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # =====================================================
    # Create Label from attack_label (if exists)
    # =====================================================
    if "attack_label" in merged.columns:
        merged["attack_label"] = merged["attack_label"].astype(str)

        # keep only target classes
        merged = merged[merged["attack_label"].isin(TARGET_CLASSES)].copy()

        label_encoder = LabelEncoder()
        label_encoder.fit(TARGET_CLASSES)

        merged["Label"] = label_encoder.transform(merged["attack_label"])

        # drop raw label columns from features
        merged = merged.drop(
            columns=["attack_label", "binary_label"],
            errors="ignore",
        )
    else:
        print(
            "WARNING: attack_label column not found. "
            "Label will not be created."
        )

    merged.to_parquet(OUTPUT_PATH, index=False)

    print(
        f"Merged dataset created with {len(merged)} samples "
        f"({len(dfs)} sources)"
    )


if __name__ == "__main__":
    merge()
