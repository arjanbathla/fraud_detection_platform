#!/usr/bin/env bash
set -e

# train once if the dataset is mounted and we don't have a champion yet
if [ -f "data/creditcard.csv" ] && [ ! -f "registry_store/index.json" ]; then
  echo "creditcard.csv found and no registry yet — training models..."
  python -m fraud_platform.train
else
  echo "skipping training (either no data/creditcard.csv, or registry already exists)"
fi

exec uvicorn fraud_platform.serving.api:app --host 0.0.0.0 --port 8000
