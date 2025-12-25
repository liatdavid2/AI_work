import os
import joblib
import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import datetime

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "model/model_with_5labels.joblib"
)

_bundle = None
_explainer = None
_global_median = None
_selected_features = None

FEATURE_DICTIONARY = {
    "flow_id": {
        "description": "Unique identifier of the network flow",
        "display": lambda v: str(int(v))
    },
    "stime": {
        "description": "When the network communication started",
        "display": lambda v: datetime.datetime.utcfromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S")
    },
    "ltime": {
        "description": "When the network communication ended",
        "display": lambda v: datetime.datetime.utcfromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S")
    },
    "dur": {
        "description": "How long the network communication lasted",
        "display": lambda v: f"{v:.3f} seconds"
    },
    "source_port": {
        "description": "Port used by the source device",
        "display": lambda v: f"port {int(v)}"
    },
    "destination_port": {
        "description": "Service port contacted on the destination",
        "display": lambda v: f"port {int(v)}"
    },
    "protocol": {
        "description": "Transport protocol used (TCP/UDP)",
        "display": lambda v: str(v)
    },
    "state": {
        "description": "Connection state of the flow",
        "display": lambda v: str(v)
    },
    "sbytes": {
        "description": "Total data sent from source to destination",
        "display": lambda v: f"{int(v)} bytes"
    },
    "dbytes": {
        "description": "Total data sent from destination to source",
        "display": lambda v: f"{int(v)} bytes"
    },
    "spkts": {
        "description": "Number of packets sent by the source",
        "display": lambda v: f"{int(v)} packets"
    },
    "dpkts": {
        "description": "Number of packets sent by the destination",
        "display": lambda v: f"{int(v)} packets"
    },
    "sload": {
        "description": "Upload rate from source",
        "display": lambda v: f"{v / 1024:.1f} KB/s"
    },
    "dload": {
        "description": "Download rate from destination",
        "display": lambda v: f"{v / 1024:.1f} KB/s"
    },
    "sttl": {
        "description": "How many network devices the packet can pass",
        "display": lambda v: f"{int(v)} hops"
    },
    "dttl": {
        "description": "TTL value observed from destination packets",
        "display": lambda v: f"{int(v)} hops"
    }
}

NON_EXPLANATORY_FEATURES = {"stime", "ltime"}


def _load_artifacts():
    global _bundle, _explainer, _global_median, _selected_features

    if _bundle is not None:
        return

    _bundle = joblib.load(MODEL_PATH)

    model = _bundle["model"]
    selector = _bundle["selector"]
    all_numeric_columns = _bundle["all_numeric_columns"]

    _explainer = shap.TreeExplainer(model)

    support_mask = selector.get_support()
    _selected_features = list(
        np.array(all_numeric_columns)[support_mask]
    )

    if "training_medians" in _bundle:
        _global_median = pd.Series(_bundle["training_medians"])
    else:
        _global_median = pd.Series(0.0, index=all_numeric_columns)


def run_inference(flow: Dict[str, Any]) -> Dict[str, Any]:
    _load_artifacts()

    model = _bundle["model"]
    scaler = _bundle["scaler"]
    selector = _bundle["selector"]
    label_encoder = _bundle["label_encoder"]
    all_numeric_columns = _bundle["all_numeric_columns"]

    df_numeric = pd.DataFrame(
        [{col: flow.get(col, np.nan) for col in all_numeric_columns}]
    ).astype(float)

    df_numeric = df_numeric.fillna(_global_median)

    df_scaled = scaler.transform(df_numeric)
    df_selected = selector.transform(df_scaled)

    pred_idx = int(model.predict(df_selected)[0])
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    shap_values = _explainer.shap_values(df_selected)
    class_idx = label_encoder.transform([pred_label])[0]

    shap_class = (
        shap_values[class_idx][0]
        if isinstance(shap_values, list)
        else shap_values[0, :, class_idx]
    )

    shap_df = pd.DataFrame({
        "feature": _selected_features,
        "value": df_numeric[_selected_features].iloc[0].values,
        "shap_value": shap_class
    })

    shap_df = shap_df[
        ~shap_df["feature"].isin(NON_EXPLANATORY_FEATURES)
    ]

    top_shap = (
        shap_df
        .assign(abs_shap=lambda d: d.shap_value.abs())
        .sort_values("abs_shap", ascending=False)
        .head(3)
    )

    explanation_parts: List[str] = []
    for _, row in top_shap.iterrows():
        direction = "supports" if row.shap_value > 0 else "goes against"
        explanation_parts.append(
            f"{row.feature}: {row.value:.3f} ({direction} '{pred_label}')"
        )

    return {
        "predicted_class": pred_label,
        "shap_top_features": top_shap.to_dict(orient="records"),
        "explanation": "; ".join(explanation_parts)
    }
