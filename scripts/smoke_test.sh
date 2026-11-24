#!/usr/bin/env bash
# Smoke test: 用极小子集验证 train -> evaluate -> predict 全链路可用（Linux/macOS/WSL）
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python"

echo "== 0. 构造 smoke 小数据集 =="
"$PY" scripts/make_smoke_data.py --categories metal_nut --n-train 12 --n-test-good 3 --n-abnormal 2

echo "== 1. 数据检查 =="
"$PY" scripts/check_data.py --category metal_nut

echo "== 2. 建库（128x128, coreset 0.25）=="
"$PY" train.py --category metal_nut --data-root data/smoke/mvtec --results-root results_smoke \
  --image-size 128 128 --coreset-ratio 0.25 --val-ratio 0.25

echo "== 3. 评估 =="
"$PY" evaluate.py --category metal_nut --results-root results_smoke --latency-samples 3 --latency-repeats 1

echo "== 4. 单张推理 =="
"$PY" predict.py --category metal_nut --results-root results_smoke --outputs-root outputs_smoke \
  --input data/smoke/mvtec/metal_nut/test/bent --no-save

echo "== 5. 批量推理 =="
"$PY" predict.py --category metal_nut --results-root results_smoke --outputs-root outputs_smoke \
  --input data/smoke/mvtec/metal_nut/test --no-save

echo "== 6. 校验产物 =="
for f in results_smoke/metal_nut/model.pt results_smoke/metal_nut/metrics.json \
         results_smoke/metal_nut/metrics.md outputs_smoke/metal_nut/results.csv; do
  [ -f "$f" ] && echo "OK  $f" || { echo "MISSING $f"; exit 1; }
done
echo "SMOKE TEST PASSED"
