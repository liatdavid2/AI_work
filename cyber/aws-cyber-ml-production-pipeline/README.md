# Production ML Pipeline for Cyber Network Traffic (Docker + AWS)

## Overview
End-to-end production ML system for network flow classification.
Provides a REST API for inference, stores predictions to S3, and supports basic monitoring and drift detection.

## Architecture
- FastAPI-based inference service
- Dockerized deployment
- AWS S3 for inference data storage
- Drift detection job (PSI-based)

## Tech Stack
- Python, FastAPI
- Scikit-learn / XGBoost
- Docker
- AWS (S3, IAM)
- Pandas, NumPy
---
## Project Structure

aws-cyber-ml-production-pipeline/
│
├── api/
│   ├── model/
│   │   ├── model_with_5labels.joblib   # Trained ML model used for inference
│   │   └── rf_features.joblib          # Feature schema aligned with the model
│   │
│   ├── api.py                          # FastAPI inference service
│   ├── graph.py                        # Inference logic and model execution flow
│   ├── __init__.py
│   ├── Dockerfile.api                  # Docker image for inference API
│   └── requirements.api.txt            # API dependencies
│
├── monitoring/
│   ├── data/                           # Reference / inference samples for drift analysis
│   ├── drift_reports/
│   │   └── drift_report_samples_*.csv  # PSI-based drift reports
│   │
│   ├── nightly_drift_job.py            # Offline drift detection job
│   ├── Dockerfile.drift                # Docker image for drift detection
│   └── requirements.drift.txt          # Drift job dependencies
│
├── training/
│   ├── data/
│   │   ├── baseline/                   # Original labeled training data (e.g. UNSW-NB15)
│   │   └── labeled_from_prod/           # Labeled samples collected from production
│   │
│   ├── data_ingestion/
│   │   └── merge_datasets.py            # Merges baseline and production-labeled data
│   │
│   ├── models/
│   │   └── train_xgb.py                 # Offline model training script
│   │
│   ├── preprocessing/
│   │   └── split.py                     # Train / validation / test split logic
│   │
│   ├── run_pipeline.py                  # End-to-end training pipeline entry point
│   ├── Dockerfile.training              # Docker image for training pipeline
│   └── requirements.training.txt        # Training dependencies
│
└── README.md                            # Project documentation
---


## Dataset
The system is based on **UNSW-NB15**, a publicly available cybersecurity dataset for network intrusion detection.

- Dataset: UNSW-NB15
- Data type: Network flow records
- Task: Multi-class attack classification (e.g., normal, generic, exploits, reconnaissance)
- Features: Packet counts, byte statistics, connection metadata, temporal and behavioral indicators

The dataset is preprocessed into structured tabular features suitable for classical ML models.

