from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime, timezone
import pandas as pd
import uuid
import os
import boto3
from io import BytesIO

from api.graph import run_inference

# =========================
# S3 configuration
# =========================
S3_BUCKET = os.getenv("INFERENCE_BUCKET", "ml-prod-inference")
S3_PREFIX = "inference"

s3 = boto3.client("s3")

app = FastAPI()


class InferenceRequest(BaseModel):
    flow: Dict[str, Any]


def save_inference_to_s3(flow: Dict[str, Any], prediction: Dict[str, Any]):
    print(">>> save_inference_to_s3 CALLED")
    record = {
        **flow,
        "predicted_class": prediction["predicted_class"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inference_id": str(uuid.uuid4())
    }

    df = pd.DataFrame([record])

    date_str = datetime.now(timezone.utc).date().isoformat()
    key = f"{S3_PREFIX}/date={date_str}/{record['inference_id']}.parquet"

    print("Attempting to upload to S3")
    print("Bucket:", S3_BUCKET)
    print("Key:", key)

    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    print(f">>> Writing to bucket={S3_BUCKET}, key={key}")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=buffer.getvalue()
    )

    print(">>> S3 put_object DONE")


@app.post("/predict")
def predict(req: InferenceRequest):
    result = run_inference(req.flow)

    save_inference_to_s3(
        flow=req.flow,
        prediction=result
    )

    return result
