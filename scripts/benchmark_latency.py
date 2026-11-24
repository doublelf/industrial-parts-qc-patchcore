#!/usr/bin/env python
"""稳定的单张推理耗时基准（含图片读取、预处理、前向、后处理）。

与 evaluate.py 内的耗时统计相比：
- 预热后对固定图片重复更多次（默认 20 张 × 5 次）；
- 输出 mean/p50/p95/p99，并落盘 results/<category>/latency_bench.json。

用法：
    python scripts/benchmark_latency.py --category metal_nut
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from qc.config import category_dir, load_config, load_run_config  # noqa: E402
from qc.device import device_description, resolve_device  # noqa: E402
from qc.logging_utils import setup_logger  # noqa: E402
from qc.metrics import latency_stats  # noqa: E402
from qc.modeling import load_checkpoint, load_image_tensor, preprocess, score_batch  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="PatchCore 单张推理耗时基准")
    p.add_argument("--config", default=None)
    p.add_argument("--category", required=True)
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--results-root", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = category_dir(args.results_root or cfg["project"]["results_root"], args.category)
    run_cfg = load_run_config(out_dir, fallback=cfg)
    log = setup_logger("benchmark", out_dir / "benchmark.log")
    device = resolve_device(run_cfg["runtime"].get("device", "auto"), log)
    image_size = tuple(run_cfg["data"]["image_size"])
    model, _ = load_checkpoint(run_cfg, args.category, out_dir / "model.pt")
    model.to(device)
    model.model.eval()

    data_root = Path(run_cfg["data"]["root"])
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()
    files = sorted((data_root / args.category / "test").rglob("*.png"))[: args.n_images]
    if not files:
        log.error("找不到测试图片: %s", data_root / args.category / "test")
        return 1

    log.info("设备=%s | 分辨率=%dx%d | 图片=%d 张 × %d 次 | 含图片读取",
             device_description(device), image_size[1], image_size[0], len(files), args.repeats)
    # 预热
    for f in files[: args.warmup]:
        score_batch(model, load_image_tensor(f).unsqueeze(0), image_size, device)

    end_to_end: list[float] = []
    forward_only: list[float] = []
    import torch

    for f in files:
        x = preprocess(load_image_tensor(f).unsqueeze(0), image_size).to(device)
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            img = load_image_tensor(f)                                  # 读图
            score_batch(model, img.unsqueeze(0), image_size, device)    # 预处理 + 前向 + 后处理
            end_to_end.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            with torch.no_grad():
                model.model(x)
            forward_only.append((time.perf_counter() - t1) * 1000.0)

    result = {
        "category": args.category,
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "device": device_description(device),
        "resolution": list(image_size),
        "includes_image_read": True,
        "n_images": len(files),
        "repeats": args.repeats,
        "end_to_end": latency_stats(end_to_end),
        "forward_only": latency_stats(forward_only),
    }
    (out_dir / "latency_bench.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    e2e, fwd = result["end_to_end"], result["forward_only"]
    log.info("端到端: mean=%.1f ms | p50=%.1f | p95=%.1f | p99=%.1f (n=%d)",
             e2e["mean_ms"], e2e["p50_ms"], e2e["p95_ms"], e2e["p99_ms"], e2e["n_samples"])
    log.info("仅前向: mean=%.1f ms | p99=%.1f ms", fwd["mean_ms"], fwd["p99_ms"])
    log.info("已保存: %s", out_dir / "latency_bench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
