# Container for the fraud detection API.
#
# The Kaggle dataset is NOT shipped in the image (licensing + size). Mount it at runtime:
#
#   docker build -t fraud-platform .
#   docker run -p 8000:8000 -v "$(pwd)/data:/app/data" fraud-platform
#
# On startup the entrypoint trains + registers the models if data/creditcard.csv is present
# and no champion exists yet, then serves the API. If the CSV is missing it still starts, but
# /predict returns 503 until you provide data and train.

FROM python:3.11-slim

# libgomp1 = OpenMP runtime that XGBoost needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["bash", "docker-entrypoint.sh"]
