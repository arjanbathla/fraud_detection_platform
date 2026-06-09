# Container for the fraud detection API.
# Build:  docker build -t fraud-platform .
# Run:    docker run -p 8000:8000 fraud-platform
#
# The image trains the models at build time so the API has a champion to serve on startup.
# (For a real deployment you'd instead bake in a registry from CI or mount it as a volume —
#  training inside the image is a convenience for this demo so `docker run` just works.)

FROM python:3.11-slim

# libgomp1 = OpenMP runtime that XGBoost needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# generate data + train + register all three models into registry_store/
RUN python -m fraud_platform.train --rows 50000 --fraud-frac 0.01

EXPOSE 8000
CMD ["uvicorn", "fraud_platform.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
