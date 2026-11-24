#!/usr/bin/env python
"""训练（建库）脚本：用 Anomalib PatchCore + MVTecAD datamodule 构建正常样本记忆库。

用法：
    python train.py --category metal_nut
    python train.py --category screw --coreset-ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from qc.config import apply_overrides, category_dir, load_config, resolve_path, snapshot_config  # noqa: E402
from qc.data_check import DOWNLOAD_HINT, require_category  # noqa: E402
from qc.device import device_description, resolve_device  # noqa: E402
from qc.logging_utils import setup_logger  # noqa: E402
from qc.metrics import select_threshold  # noqa: E402
from qc.modeling import (  # noqa: E402
    build_datamodule,
    build_engine,
    build_model,
    save_checkpoint,
    score_batch,
)
from qc.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchCore 建库训练")
    p.add_argument("--config", default=None, help="配置文件路径（默认 config.yaml）")
    p.add_argument("--category", required=True, help="MVTec 类别，例如 metal_nut / screw")
    p.add_argument("--coreset-ratio", type=float, default=None, help="覆盖 coreset 采样比例")
    p.add_argument("--backbone", default=None, help="覆盖 backbone")
    p.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"), help="覆盖输入尺寸")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None, help="覆盖设备策略")
    p.add_argument("--data-root", default=None, help="覆盖数据根目录（smoke test 用）")
    p.add_argument("--results-root", default=None, help="覆盖结果根目录（smoke test 用）")
    p.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    p.add_argument("--val-ratio", type=float, default=None, help="覆盖验证集比例（从 train/good 切分）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {
        "model.coreset_sampling_ratio": args.coreset_ratio,
        "model.backbone": args.backbone,
        "data.image_size": args.image_size,
        "data.root": args.data_root,
        "project.results_root": args.results_root,
        "runtime.device": args.device,
        "project.seed": args.seed,
        "data.val_split.ratio": args.val_ratio,
    }
    apply_overrides(cfg, overrides)
    category = args.category
    out_dir = category_dir(cfg["project"]["results_root"], category)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger("train", out_dir / "train.log")

    log.info("=" * 74)
    log.info("PatchCore 建库开始 | 类别=%s | 配置文件=%s", category, cfg["_config_path"])
    log.info("=" * 74)

    # 1) 数据检查（缺失时给下载提示而不是裸异常）
    try:
        stats = require_category(cfg["data"]["root"], category)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    log.info("数据: train/good=%d | test 正常=%d | test 异常=%d | 缺陷类型=%s",
             stats.train_good, stats.test_counts.get("good", 0), stats.test_abnormal,
             [k for k in stats.test_counts if k != "good"])
    for w in stats.warnings:
        log.warning("数据告警: %s", w)

    # 2) 设备与随机种子
    device = resolve_device(cfg["runtime"].get("device", "auto"), log)
    cfg["runtime"]["resolved_device"] = str(device)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    log.info("随机种子=%d | 设备=%s | 输入尺寸=%s", seed, device_description(device), cfg["data"]["image_size"])

    # 3) datamodule + 模型
    from qc.modeling import _anomalib_version

    log.info("Anomalib=%s | backbone=%s | layers=%s | coreset_ratio=%.3f",
             _anomalib_version(), cfg["model"]["backbone"], cfg["model"]["layers"],
             float(cfg["model"]["coreset_sampling_ratio"]))
    dm = build_datamodule(cfg, category)
    dm.setup()
    n_train = len(dm.train_data)
    n_val = len(dm.val_data) if getattr(dm, "val_data", None) is not None else 0
    n_test = len(dm.test_data)
    log.info("官方划分: 拟合用正常样本=%d | 验证用正常样本=%d | 测试集=%d（test 未参与建库）",
             n_train, n_val, n_test)

    model = build_model(cfg)
    engine = build_engine(cfg, default_root_dir=out_dir / "engine")

    # 4) 建库（Anomalib Engine 调用 PatchCoreMemoryBank 流程）
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    fit_seconds = time.perf_counter() - t0
    mb_shape = tuple(model.model.memory_bank.shape)
    log.info("建库完成: 耗时=%.1fs | 记忆库形状=%s | 记忆库大小=%.1f MB",
             fit_seconds, mb_shape, model.model.memory_bank.numel() * 4 / 1024**2)

    # Lightning 会自动保存一份 checkpoint（含完整 backbone，数百 MB）；
    # 我们随后会保存自己的精简 checkpoint，这里清理 Engine 产物避免重复占盘。
    engine_dir = out_dir / "engine"
    if engine_dir.exists():
        for ck in engine_dir.rglob("*.ckpt"):
            try:
                ck.unlink()
            except OSError:
                pass

    # 5) 验证集打分（正常样本），确定阈值
    model.model.eval()
    val_scores: list[float] = []
    t1 = time.perf_counter()
    with np.errstate(all="ignore"):
        for batch in dm.val_dataloader():
            images = batch["image"]
            scores, _ = score_batch(model, images, cfg["data"]["image_size"], device)
            val_scores.extend(scores.tolist())
    val_seconds = time.perf_counter() - t1
    log.info("验证集打分完成: %d 张 | 耗时=%.1fs", len(val_scores), val_seconds)

    thr_cfg = cfg["threshold"]
    threshold_info = select_threshold(
        val_scores,
        thr_cfg["strategy"],
        float(thr_cfg.get("quantile", 0.99)),
        float(thr_cfg.get("k", 3.0)),
    )
    log.info("阈值: %.6f | 策略=%s | 分位=%.3f | 验证集正常样本误报率=%.4f",
             threshold_info["threshold"], threshold_info["strategy"],
             threshold_info["quantile"] or float("nan"), threshold_info["val_fp_rate"])

    # 6) 保存模型、阈值与本次配置
    ckpt_path = out_dir / "model.pt"
    save_checkpoint(model, cfg, category, ckpt_path, threshold_info)
    log.info("模型已保存: %s (%.1f MB)", ckpt_path, ckpt_path.stat().st_size / 1024**2)

    (out_dir / "threshold.json").write_text(json.dumps(threshold_info, indent=2, ensure_ascii=False), encoding="utf-8")
    snapshot_config(cfg, out_dir / "config.yaml")

    run_info = {
        "category": category,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "device": device_description(device),
        "image_size": list(cfg["data"]["image_size"]),
        "backbone": cfg["model"]["backbone"],
        "layers": list(cfg["model"]["layers"]),
        "coreset_sampling_ratio": float(cfg["model"]["coreset_sampling_ratio"]),
        "num_neighbors": int(cfg["model"]["num_neighbors"]),
        "n_fit": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "memory_bank_shape": list(mb_shape),
        "fit_seconds": fit_seconds,
        "val_score_seconds": val_seconds,
        "threshold": threshold_info,
        "anomalib_version": _anomalib_version(),
        "data_counts": {
            "train_good": stats.train_good,
            "test_good": stats.test_counts.get("good", 0),
            "test_abnormal": stats.test_abnormal,
        },
        "val_scores": sorted(val_scores),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("产物目录: %s", out_dir)
    log.info("下一步: python evaluate.py --category %s", category)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(exc)
        print(DOWNLOAD_HINT)
        raise SystemExit(1) from None
