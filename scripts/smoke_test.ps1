# Smoke test (Windows PowerShell)
# 用极小子集验证 train -> evaluate -> predict 全链路可用。
# 用法: pwsh -File scripts/smoke_test.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "== 0. 构造 smoke 小数据集 =="
& $py scripts\make_smoke_data.py --categories metal_nut --n-train 12 --n-test-good 3 --n-abnormal 2

Write-Host "== 1. 数据检查 =="
& $py scripts\check_data.py --category metal_nut

Write-Host "== 2. 建库 (128x128, coreset 0.25) =="
& $py train.py --category metal_nut --data-root data/smoke/mvtec --results-root results_smoke `
    --image-size 128 128 --coreset-ratio 0.25 --val-ratio 0.25
if ($LASTEXITCODE -ne 0) { throw "train.py failed" }

Write-Host "== 3. 评估 =="
& $py evaluate.py --category metal_nut --results-root results_smoke --latency-samples 3 --latency-repeats 1
if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed" }

Write-Host "== 4. 单张推理 =="
& $py predict.py --category metal_nut --results-root results_smoke --outputs-root outputs_smoke `
    --input data\smoke\mvtec\metal_nut\test\bent --no-save
if ($LASTEXITCODE -ne 0) { throw "predict.py (single) failed" }

Write-Host "== 5. 批量推理 =="
& $py predict.py --category metal_nut --results-root results_smoke --outputs-root outputs_smoke `
    --input data\smoke\mvtec\metal_nut\test --no-save
if ($LASTEXITCODE -ne 0) { throw "predict.py (batch) failed" }

Write-Host "== 6. 校验产物 =="
$required = @(
    "results_smoke\metal_nut\model.pt",
    "results_smoke\metal_nut\metrics.json",
    "results_smoke\metal_nut\metrics.md",
    "results_smoke\metal_nut\threshold.json",
    "outputs_smoke\metal_nut\results.csv"
)
foreach ($f in $required) {
    if (-not (Test-Path $f)) { throw "缺少产物: $f" }
    Write-Host "OK  $f"
}
Write-Host "SMOKE TEST PASSED"
