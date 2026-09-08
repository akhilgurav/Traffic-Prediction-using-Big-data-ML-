# run_pipeline_no_train.ps1
# Rebuilds and restarts the full pipeline WITHOUT re-running feature
# engineering / labeling / training - reuses whatever model is already
# saved at data/models/congestion_rf/. Use this for normal day-to-day
# restarts; use run_pipeline.ps1 only when you actually want to retrain.
# Run from the project root: .\run_pipeline_no_train.ps1

$ErrorActionPreference = "Stop"

$ModelPath = "data/models/congestion_rf/metrics.json"
if (-not (Test-Path $ModelPath)) {
    Write-Host "WARNING: no trained model found at $ModelPath" -ForegroundColor Yellow
    Write-Host "Streaming/dashboard will start anyway, but predictions and the" -ForegroundColor Yellow
    Write-Host "Model Performance page won't have anything to show yet." -ForegroundColor Yellow
    Write-Host "Run .\run_pipeline.ps1 at least once first to train a model." -ForegroundColor Yellow
}

Write-Host "==> Starting core stack + kafka + hdfs + streaming + dashboard profiles" -ForegroundColor Cyan
docker compose --profile kafka --profile hdfs --profile streaming --profile dashboard up --build -d

Write-Host "==> Waiting for postgres to be healthy" -ForegroundColor Cyan
docker compose exec postgres pg_isready -U traffic_user -d traffic_db
Start-Sleep -Seconds 5

Write-Host "==> Row count so far" -ForegroundColor Cyan
docker compose exec postgres psql -U traffic_user -d traffic_db -c "SELECT count(*) FROM traffic_snapshots;"

Write-Host "==> Pipeline up (model NOT retrained). Tailing collector logs (Ctrl+C to stop watching, containers keep running)" -ForegroundColor Green
docker compose logs -f collector
