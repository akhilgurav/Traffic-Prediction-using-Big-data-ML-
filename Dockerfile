FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2 build (binary wheel usually avoids this, kept for safety)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ is where raw JSON + parquet land — mounted as a volume in compose
RUN mkdir -p data/raw_json data/parquet

CMD ["python", "collector_main.py"]
