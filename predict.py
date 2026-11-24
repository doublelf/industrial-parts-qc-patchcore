#!/usr/bin/env python
"""单张 / 批量推理脚本。

用法：
    python predict.py --category metal_nut --input path/to/image.png
    python predict.py --category metal_nut --input path/to/folder
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from qc.config import (  # noqa: E402
    category_dir,
    load_config,
    load_run_config,
    outputs_dir,
)
from qc.device import device_description, resolve_device  # noqa: E402
from qc.logging_utils import setup_logger  # noqa: E402
from qc.modeling import load_checkpoint, load_image_tensor, score_batch  # noqa: E402
from qc.seed import set_seed  # noqa: E402
from qc.viz import save_prediction_overlay  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchCore 单张/批量推理")
    p.add_argument("--config", default=None)
    p.add_argument("--category", required=True)
    p.add_argument("--input", required=True, help="图片路径或文件夹")
    p.add_argument("--results-root", default=None)
    p.add_argument("--outputs-root", default=None, help="覆盖叠加图输出根目录")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--threshold", type=float, default=None, help="手动覆盖阈值（仅用于调试，训练得到的阈值仍以 threshold.json 为准）")
    p.add_argument("--no-save", action="store_true", help="只输出判定，不保存叠加图")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    results_root = args.results_root or cfg["project"]["results_root"]
    out_dir = category_dir(results_root, args.category)
    if not (out_dir / "model.pt").exists():
        print(f"未找到模型: {out_dir / 'model.pt'}\n请先运行: python train.py --category {args.category}")
        return 1
    run_cfg = load_run_config(out_dir, fallback=cfg)
    if args.device:
        run_cfg["runtime"]["device"] = args.device
    device = resolve_device(run_cfg["runtime"].get("device", "auto"), None)
    run_cfg["runtime"]["resolved_device"] = str(device)
    set_seed(int(run_cfg["project"]["seed"]))

    out_root = outputs_dir(args.outputs_root or cfg["project"]["outputs_root"], args.category)
    out_root.mkdir(parents=True, exist_ok=True)
    log = setup_logger("predict", out_root / "predict.log")

    thr_path = out_dir / "threshold.json"
    if not thr_path.exists():
        log.error("缺少阈值文件: %s，请先运行 train.py", thr_path)
        return 1
    thr_info = json.loads(thr_path.read_text(encoding="utf-8"))
    threshold = float(args.threshold if args.threshold is not None else thr_info["threshold"])
    if args.threshold is not None:
        log.warning("使用手动阈值 %.6f（覆盖验证集阈值 %.6f），仅用于调试", args.threshold, thr_info["threshold"])

    model, _ = load_checkpoint(run_cfg, args.category, out_dir / "model.pt")
    model.to(device)
    model.model.eval()
    image_size = tuple(run_cfg["data"]["image_size"])

    src = Path(args.input)
    if not src.exists():
        log.error("输入路径不存在: %s", src)
        return 1
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
        if not files:
            log.error("目录中没有图片: %s", src)
            return 1
        return run_batch(files, model, image_size, device, threshold, out_root, log, args.no_save)
    return run_single(src, model, image_size, device, threshold, out_root, log, args.no_save)


def _infer_one(path: Path, model, image_size, device, threshold: float, out_root: Path,
               save: bool) -> dict:
    t0 = time.perf_counter()
    img = load_image_tensor(path)
    scores, maps = score_batch(model, img.unsqueeze(0), image_size, device)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    score = float(scores[0])
    decision = "NG" if score >= threshold else "OK"

    overlay_path = ""
    if save:
        from PIL import Image

        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        p = save_prediction_overlay(out_root, path, rgb, maps[0, 0].numpy(), score, decision, threshold)
        overlay_path = str(p)
    return {
        "file": path.name,
        "score": score,
        "decision": decision,
        "latency_ms": latency_ms,
        "overlay": overlay_path,
        "threshold": threshold,
        "path": str(path),
    }


def run_single(path: Path, model, image_size, device, threshold, out_root, log, no_save) -> int:
    res = _infer_one(path, model, image_size, device, threshold, out_root, not no_save)
    log.info("=" * 60)
    log.info("图片: %s", res["path"])
    log.info("异常分数: %.6f | 阈值: %.6f | 判定: %s", res["score"], res["threshold"], res["decision"])
    log.info("耗时(含读取): %.1f ms | 设备: %s | 分辨率: %s", res["latency_ms"], device_description(device), image_size)
    if res["overlay"]:
        log.info("热力图叠加图: %s", res["overlay"])
    log.info("=" * 60)
    print(json.dumps({k: res[k] for k in ("file", "score", "decision", "latency_ms", "threshold", "overlay")},
                     ensure_ascii=False))
    return 0


def run_batch(files: list[Path], model, image_size, device, threshold, out_root, log, no_save) -> int:
    log.info("批量推理: %d 张图片 | 设备=%s | 阈值=%.6f", len(files), device_description(device), threshold)
    rows = []
    t_all = time.perf_counter()
    for i, p in enumerate(files, 1):
        res = _infer_one(p, model, image_size, device, threshold, out_root, not no_save)
        rows.append(res)
        if i % 10 == 0 or i == len(files):
            log.info("  进度 %d/%d | 最近: %s %.4f %s", i, len(files), res["file"], res["score"], res["decision"])
    total = time.perf_counter() - t_all

    csv_path = out_root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "分数", "判定", "耗时_ms"])
        for r in rows:
            w.writerow([r["file"], f"{r['score']:.6f}", r["decision"], f"{r['latency_ms']:.1f}"])
    n_ng = sum(1 for r in rows if r["decision"] == "NG")
    log.info("批量推理完成: %d 张 | NG=%d | OK=%d | 总耗时=%.1fs | 平均=%.1f ms/张",
             len(rows), n_ng, len(rows) - n_ng, total, total / len(rows) * 1000)
    log.info("CSV: %s", csv_path)
    (out_root / "results.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "category": out_root.name,
        "threshold": threshold,
        "n_images": len(rows),
        "n_ng": n_ng,
        "inputs": [r["path"] for r in rows],
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from None
