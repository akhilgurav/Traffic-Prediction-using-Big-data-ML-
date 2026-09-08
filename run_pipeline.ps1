# run_pipeline.ps1
# Rebuilds and restarts the full Pune traffic pipeline in one go:
# core stack -> kafka -> hdfs -> feature engineering -> labeling ->
# training -> streaming -> dashboard.
# Run from the project root: .\run_pipeline.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> Starting core stack + kafka + hdfs + dashboard profiles" -ForegroundColor Cyan
docker compose --profile kafka --profile hdfs --profile dashboard up --build -d

Write-Host "==> Waiting for postgres to be healthy" -ForegroundColor Cyan
docker compose exec postgres pg_isready -U traffic_user -d traffic_db
Start-Sleep -Seconds 5

Write-Host "==> Row count so far" -ForegroundColor Cyan
docker compose exec postgres psql -U traffic_user -d traffic_db -c "SELECT count(*) FROM traffic_snapshots;"

Write-Host "==> Feature engineering" -ForegroundColor Cyan
docker compose --profile spark run --rm spark spark-submit /app/spark_jobs/feature_engineering.py

Write-Host "==> Label generation" -ForegroundColor Cyan
docker compose --profile spark run --rm spark spark-submit /app/spark_jobs/label_generation.py

Write-Host "==> Model training" -ForegroundColor Cyan
docker compose --profile spark run --rm spark spark-submit /app/spark_jobs/train_model.py

docker compose --profile spark run --rm spark spark-submit /app/spark_jobs/streaming_predict.py

Write-Host "==> Starting streaming inference (spark-streaming)" -ForegroundColor Cyan
docker compose --profile kafka --profile streaming up -d spark-streaming

Write-Host "==> Pipeline fully up. Tailing collector logs (Ctrl+C to stop watching, containers keep running)" -ForegroundColor Green
docker compose logs -f collector
