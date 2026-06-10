#!/usr/bin/env bash
set -e

# build the SQLite db from the CSV if it isn't there yet
if [ -f "data/creditcard.csv" ] && [ ! -f "data/creditcard.db" ]; then
  echo "creditcard.csv found, no db yet — loading into SQLite..."
  python -m fraud_platform.data.load_to_sqlite
fi

# train once if the db exists and we don't have a champion yet
if [ -f "data/creditcard.db" ] && [ ! -f "registry_store/index.json" ]; then
  echo "db ready and no registry yet — training models..."
  python -m fraud_platform.train
else
  echo "skipping training (either no data/creditcard.db, or registry already exists)"
fi

exec uvicorn fraud_platform.serving.api:app --host 0.0.0.0 --port 8000
