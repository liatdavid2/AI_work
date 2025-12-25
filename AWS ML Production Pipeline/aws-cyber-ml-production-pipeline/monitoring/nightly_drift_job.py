import os
import pandas as pd
from io import BytesIO
import numpy as np
import boto3
from datetime import datetime, timezone
from typing import List, Optional

# =====================================================
# Configuration
# =====================================================

MIN_SAMPLES = 100
LOW_CONFIDENCE_SAMPLES = 1000
EPS = 1e-6

REFERENCE_DATA_PATH = "data/UNSW_Flow.parquet"

S3_BUCKET = os.getenv(
    "INFERENCE_BUCKET",
    "ml-prod-inference"
)

S3_PREFIX = os.getenv(
    "INFERENCE_PREFIX",
    "inference"
)

DRIFT_OUTPUT_DIR = os.getenv(
    "DRIFT_OUTPUT_DIR",
    "drift_reports"
)

N_BINS = int(os.getenv("PSI_BINS", 10))

os.makedirs(DRIFT_OUTPUT_DIR, exist_ok=True)

s3 = boto3.client("s3")

# =====================================================
# PSI implementation
# =====================================================

def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    n_bins: int
) -> Optional[float]:
    """
    Population Stability Index (PSI)

    Returns:
        float: PSI value
        None : if PSI should not be computed (insufficient data)

    PSI interpretation:
        < 0.1   -> no drift
        0.1-0.25 -> moderate drift
        > 0.25  -> significant drift
    """

    expected = expected.dropna()
    actual = actual.dropna()

    # Guard: insufficient inference samples
    if len(actual) < MIN_SAMPLES:
        return None

    if expected.empty or actual.empty:
        return None

    quantiles = np.linspace(0, 1, n_bins + 1)
    breakpoints = np.unique(expected.quantile(quantiles).values)

    # Not enough unique bins
    if len(breakpoints) < 2:
        return None

    expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True)
    actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True)

    expected_dist = expected_bins.value_counts(normalize=True)
    actual_dist = actual_bins.value_counts(normalize=True)

    psi = 0.0

    for bucket in expected_dist.index:
        e = expected_dist.get(bucket, EPS)
        a = actual_dist.get(bucket, EPS)

        # Numerical stability protection
        psi += (a - e) * np.log((a + EPS) / (e + EPS))

    return float(psi)


def psi_category(psi: Optional[float]) -> str:
    if psi is None:
        return "insufficient_data"
    if psi < 0.1:
        return "no_drift"
    if psi < 0.25:
        return "moderate_drift"
    return "significant_drift"


def confidence_level(n_samples: int) -> str:
    if n_samples < MIN_SAMPLES:
        return "INSUFFICIENT_DATA"
    if n_samples < LOW_CONFIDENCE_SAMPLES:
        return "LOW_CONFIDENCE_DRIFT"
    return "NORMAL_PSI"


# =====================================================
# Load inference parquet files from S3 (daily)
# =====================================================

def load_daily_inference_from_s3(date_str: str) -> pd.DataFrame:
    """
    Loads all parquet inference files for a given date:
    s3://bucket/inference/date=YYYY-MM-DD/*.parquet
    """

    prefix = f"{S3_PREFIX}/date={date_str}/"

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)

    parquet_keys = []

    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                parquet_keys.append(obj["Key"])

    if not parquet_keys:
        return pd.DataFrame()

    dfs = []

    for key in parquet_keys:
        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        body = response["Body"].read()
        df = pd.read_parquet(BytesIO(body))
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# =====================================================
# Main nightly drift job
# =====================================================

def run_nightly_drift_job():
    print("Starting nightly drift job")

    reference_df = pd.read_parquet(REFERENCE_DATA_PATH)

    today = datetime.now(timezone.utc).date().isoformat()
    inference_df = load_daily_inference_from_s3(today)

    reference_size = len(reference_df)
    inference_size = len(inference_df)

    print(f"Reference samples: {reference_size}")
    print(f"Inference samples: {inference_size}")

    # Global guard: no inference data
    if inference_size == 0:
        print("No inference data found for today. Skipping drift computation.")
        return

    excluded_cols = {
        "predicted_class",
        "timestamp",
        "inference_id"
    }

    common_features: List[str] = [
        c for c in reference_df.columns
        if c in inference_df.columns and c not in excluded_cols
    ]

    results = []

    for feature in common_features:
        psi = calculate_psi(
            reference_df[feature],
            inference_df[feature],
            N_BINS
        )

        results.append({
            "feature": feature,
            "psi": None if psi is None else round(psi, 4),
            "drift_level": psi_category(psi),
            "confidence": confidence_level(inference_size),
            "reference_samples": reference_size,
            "inference_samples": inference_size
        })

    drift_df = (
        pd.DataFrame(results)
        .sort_values(
            by="psi",
            ascending=False,
            na_position="last"
        )
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_name = f"drift_report_samples_{inference_size}_{timestamp}.csv"
    output_path = os.path.join(DRIFT_OUTPUT_DIR, output_name)

    drift_df.to_csv(output_path, index=False)

    print(f"Drift report written to {output_path}")

    print(
        drift_df["drift_level"]
        .value_counts()
        .to_string()
    )


# =====================================================
# Entrypoint
# =====================================================

if __name__ == "__main__":
    run_nightly_drift_job()
