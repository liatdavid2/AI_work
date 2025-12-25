import subprocess

steps = [
    "training/data_ingestion/merge_datasets.py",
    "training/preprocessing/split.py",
    "training/models/train_xgb.py"
]

for step in steps:
    print(f"Running: {step}")
    subprocess.run(["python", step], check=True)

print("Training pipeline completed successfully")
