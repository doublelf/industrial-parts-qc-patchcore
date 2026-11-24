#!/usr/bin/env python
"""评估脚本：在官方 test 集上计算指标，并生成 metrics.json / metrics.md / 可视化。

用法：
    python evaluate.py --category metal_nut
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from qc.config import category_dir, load_config, load_run_config, snapshot_config  # noqa: E402
from qc.data_check import require_category  # noqa: E402
from qc.device import device_description, resolve_device  # noqa: E402
from qc.logging_utils import setup_logger  # noqa: E402
from qc.metrics import (  # noqa: E402
    compute_aupro,
    image_metrics,
    latency_stats,
    pixel_metrics,
    summarize_per_type,
    threshold_confusion,
)
from qc.modeling import (  # noqa: E402
    _anomalib_version,
    build_datamodule,
    load_checkpoint,
    load_image_tensor,
    score_batch,
)
from qc.seed import set_seed  # noqa: E402
from qc.viz import make_comparison_grid, save_prediction_overlay  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchCore 评估")
    p.add_argument("--config", default=None, help="配置文件路径（默认 config.yaml）")
    p.add_argument("--category", required=True)
    p.add_argument("--results-root", default=None, help="覆盖结果根目录")
    p.add_argument("--data-root", default=None, help="覆盖数据根目录")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--latency-samples", type=int, default=20, help="耗时统计使用的图片数量上限")
    p.add_argument("--latency-repeats", type=int, default=3, help="每张图重复计时次数")
    p.add_argument("--no-vis", action="store_true", help="跳过可视化（快速冒烟）")
    return p.parse_args()


def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """把 GT 掩码缩放到异常图尺寸（最近邻，保持 0/1）。"""
    m = mask.float().unsqueeze(0).unsqueeze(0)
    m = torch.nn.functional.interpolate(m, size=size, mode="nearest")
    return (m[0, 0].numpy() > 0.5).astype(np.uint8)


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


@torch.no_grad()
def _measure_latency(model, paths: list[str], image_size, device, repeats: int) -> list[float]:
    times_ms: list[float] = []
    for p in paths:
        for _ in range(repeats):
            t0 = time.perf_counter()
            img = load_image_tensor(p)                      # 含图片读取
            score_batch(model, img.unsqueeze(0), image_size, device)  # 预处理 + 前向 + 后处理
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    results_root = args.results_root or cfg["project"]["results_root"]
    out_dir = category_dir(results_root, args.category)
    if not out_dir.exists():
        print(f"未找到训练产物目录: {out_dir}\n请先运行: python train.py --category {args.category}")
        return 1
    log = setup_logger("evaluate", out_dir / "evaluate.log")

    run_cfg = load_run_config(out_dir, fallback=cfg)
    if args.data_root:
        run_cfg["data"]["root"] = args.data_root
    if args.device:
        run_cfg["runtime"]["device"] = args.device
    device = resolve_device(run_cfg["runtime"].get("device", "auto"), log)
    run_cfg["runtime"]["resolved_device"] = str(device)
    set_seed(int(run_cfg["project"]["seed"]))

    log.info("=" * 74)
    log.info("PatchCore 评估 | 类别=%s | 设备=%s | 输入尺寸=%s", args.category,
             device_description(device), run_cfg["data"]["image_size"])
    log.info("=" * 74)

    try:
        stats = require_category(run_cfg["data"]["root"], args.category)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    thr_path = out_dir / "threshold.json"
    if not thr_path.exists():
        log.error("缺少阈值文件 %s，请先运行 train.py", thr_path)
        return 1
    thr_info = json.loads(thr_path.read_text(encoding="utf-8"))
    threshold = float(thr_info["threshold"])
    log.info("阈值来自验证集: %.6f | 策略=%s | 验证集正常样本数=%d | 验证集误报率=%.4f",
             threshold, thr_info["strategy"], thr_info["n_val"], thr_info["val_fp_rate"])

    model, payload = load_checkpoint(run_cfg, args.category, out_dir / "model.pt")
    model.to(device)
    model.model.eval()
    log.info("模型已加载: 记忆库=%s", tuple(model.model.memory_bank.shape))

    dm = build_datamodule(run_cfg, args.category)
    dm.setup()
    image_size = tuple(run_cfg["data"]["image_size"])
    map_size = (int(image_size[0]), int(image_size[1]))

    labels: list[int] = []
    scores: list[float] = []
    types: list[str] = []
    paths: list[str] = []
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    t0 = time.perf_counter()
    for batch in dm.test_dataloader():
        images = batch["image"]
        s, m = score_batch(model, images, image_size, device)
        bs = len(images)
        labels.extend([int(x) for x in batch["gt_label"].tolist()])
        scores.extend(s.tolist())
        for i in range(bs):
            p = str(batch["image_path"][i])
            paths.append(p)
            types.append(Path(p).parent.name)
            maps.append(m[i, 0].numpy().astype(np.float32))
            masks.append(_resize_mask(batch["gt_mask"][i], map_size))
    infer_seconds = time.perf_counter() - t0
    log.info("测试集推理完成: %d 张 | 耗时=%.1fs | 平均=%.1f ms/张（含读取与预处理）",
             len(scores), infer_seconds, infer_seconds / max(len(scores), 1) * 1000)

    # ---------------- 指标 ----------------
    img_m = image_metrics(labels, scores)
    pix_labels = np.concatenate([m.reshape(-1) for m in masks])
    pix_scores = np.concatenate([m.reshape(-1) for m in maps])
    pix_m = pixel_metrics(pix_labels, pix_scores)
    aupro = compute_aupro(maps, masks, fpr_limit=float(run_cfg["evaluation"].get("aupro_fpr_limit", 0.3)))
    conf = threshold_confusion(labels, scores, threshold)
    per_type = summarize_per_type(labels, scores, types, threshold)

    log.info("图像级 AUROC=%.4f | 图像级 AP=%.4f", img_m["image_auroc"], img_m["image_ap"])
    log.info("像素级 AUROC=%.4f | 像素级 AP=%.4f", pix_m["pixel_auroc"], pix_m["pixel_ap"])
    log.info("AU-PRO=%s", "不支持" if aupro is None else f"{aupro:.4f}")
    log.info("阈值=%.6f -> 误报 FP=%d | 漏检 FN=%d | Precision=%.4f | Recall=%.4f",
             threshold, conf["fp"], conf["fn"], conf["precision"], conf["recall"])

    # ---------------- 耗时 ----------------
    lat_paths = paths[: max(1, args.latency_samples)]
    lat_ms = _measure_latency(model, lat_paths, image_size, device, max(1, args.latency_repeats))
    lat = latency_stats(lat_ms)
    log.info("单张耗时: mean=%.1f ms | p99=%.1f ms | n=%d | 设备=%s | 分辨率=%dx%d | 含图片读取",
             lat["mean_ms"], lat["p99_ms"], lat["n_samples"], device_description(device), image_size[1], image_size[0])

    # ---------------- 失败案例 ----------------
    order = np.argsort(scores)
    missed = [i for i in order if labels[i] == 1][:5]           # 分数最低的异常：漏检
    false_alarm = [i for i in order[::-1] if labels[i] == 0][:5]  # 分数最高的正常：误报
    failures = {
        "top_missed_abnormal": [
            {"file": paths[i], "type": types[i], "score": float(scores[i])} for i in missed[:3]
        ],
        "top_false_alarm_normal": [
            {"file": paths[i], "type": types[i], "score": float(scores[i])} for i in false_alarm[:3]
        ],
    }

    # ---------------- 可视化 ----------------
    vis_dir = out_dir / "visualizations"
    if not args.no_vis:
        rng = random.Random(int(run_cfg["project"]["seed"]))
        abnormal_idx = [i for i, y in enumerate(labels) if y == 1]
        normal_idx = [i for i, y in enumerate(labels) if y == 0]
        n_ab = min(int(run_cfg["evaluation"].get("n_abnormal_vis", 6)), len(abnormal_idx))
        n_no = min(int(run_cfg["evaluation"].get("n_normal_vis", 3)), len(normal_idx))
        pick_ab = rng.sample(abnormal_idx, n_ab)
        pick_no = rng.sample(normal_idx, n_no)

        def _sample(i: int) -> dict:
            img = _load_rgb(paths[i])
            mask = _load_rgb_mask(paths[i]) if labels[i] == 1 else None
            decision = "NG" if scores[i] >= threshold else "OK"
            return {
                "image": img,
                "mask": mask,
                "map": maps[i],
                "score": float(scores[i]),
                "decision": decision,
                "name": f"{types[i]}/{Path(paths[i]).name}",
                "label": int(labels[i]),
            }

        samples = [_sample(i) for i in pick_ab + pick_no]
        make_comparison_grid(samples, vis_dir / "comparison_grid.png",
                             title=f"{args.category} | PatchCore | 6 abnormal + 3 normal (seed={run_cfg['project']['seed']})")
        # 同时保存逐张叠加图，方便查看
        for s in samples:
            save_prediction_overlay(vis_dir / "overlay", Path(s["name"].split("/")[-1]),
                                    s["image"], s["map"], s["score"], s["decision"], threshold)
        log.info("可视化已保存: %s", vis_dir / "comparison_grid.png")

    # ---------------- 落盘 ----------------
    metrics = {
        "category": args.category,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(run_cfg["project"]["seed"]),
        "device": device_description(device),
        "image_size": list(image_size),
        "anomalib_version": _anomalib_version(),
        "n_test": len(scores),
        "n_normal": int(sum(1 for y in labels if y == 0)),
        "n_abnormal": int(sum(1 for y in labels if y == 1)),
        "image_auroc": img_m["image_auroc"],
        "image_ap": img_m["image_ap"],
        "pixel_auroc": pix_m["pixel_auroc"],
        "pixel_ap": pix_m["pixel_ap"],
        "aupro": aupro,
        "aupro_supported": aupro is not None,
        "aupro_fpr_limit": float(run_cfg["evaluation"].get("aupro_fpr_limit", 0.3)),
        "threshold": conf,
        "threshold_source": thr_info,
        "latency": {**lat, "includes_image_read": True, "resolution": list(image_size),
                    "device": device_description(device), "n_images": len(lat_paths),
                    "repeats": int(args.latency_repeats)},
        "per_type": per_type,
        "failures": failures,
        "per_image": [
            {"file": str(p), "type": t, "label": int(y), "score": float(s),
             "decision": "NG" if s >= threshold else "OK"}
            for p, t, y, s in zip(paths, types, labels, scores)
        ],
        "test_inference_seconds": infer_seconds,
        "data_counts": {
            "train_good": stats.train_good,
            "test_good": stats.test_counts.get("good", 0),
            "test_abnormal": stats.test_abnormal,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metrics.md").write_text(_render_markdown(metrics), encoding="utf-8")
    snapshot_config(run_cfg, out_dir / "config.yaml")  # 记录本次评估实际使用的配置
    log.info("指标已保存: %s 和 %s", out_dir / "metrics.json", out_dir / "metrics.md")
    return 0


def _load_rgb_mask(image_path: str) -> np.ndarray | None:
    """按 MVTec 命名规则找 GT 掩码：ground_truth/<type>/<stem>_mask.png。"""
    p = Path(image_path)
    parts = list(p.parts)
    try:
        i = parts.index("test")
    except ValueError:
        return None
    parts[i] = "ground_truth"
    parts[-1] = p.stem + "_mask.png"
    mask_path = Path(*parts)
    if not mask_path.exists():
        return None
    with Image.open(mask_path) as im:
        return (np.asarray(im.convert("L")) > 127).astype(np.uint8)


def _render_markdown(m: dict) -> str:
    aupro = f"{m['aupro']:.4f}" if m["aupro_supported"] else "不支持（Anomalib 未返回结果）"
    lines = [
        f"# PatchCore 评估报告 — {m['category']}",
        "",
        f"- 评估时间：{m['evaluated_at']}",
        f"- 设备：{m['device']}",
        f"- 输入尺寸：{m['image_size'][0]}x{m['image_size'][1]}",
        f"- 随机种子：{m['seed']}",
        f"- Anomalib：{m['anomalib_version']}",
        f"- 测试集：{m['n_test']} 张（正常 {m['n_normal']} / 异常 {m['n_abnormal']}），官方 test 划分",
        "",
        "## 1. 核心指标（官方 test 集）",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 图像级 AUROC | **{m['image_auroc']:.4f}** |",
        f"| 图像级 AP | **{m['image_ap']:.4f}** |",
        f"| 像素级 AUROC | **{m['pixel_auroc']:.4f}** |",
        f"| 像素级 AP | **{m['pixel_ap']:.4f}** |",
        f"| AU-PRO（FPR≤{m['aupro_fpr_limit']}） | **{aupro}** |",
        "",
        "> 阈值只使用验证集（官方 train/good 切分出的正常样本）确定，未在 test 上调参。",
        "",
        "## 2. 阈值与混淆矩阵",
        "",
        f"- 阈值来源：{m['threshold_source']['strategy']}"
        + (f"，k={m['threshold_source'].get('k')}" if m["threshold_source"].get("k") is not None else
           f"，分位数 {m['threshold_source'].get('quantile')}")
        + f"，验证集 {m['threshold_source']['n_val']} 张正常样本",
        f"- 验证集分数：median={m['threshold_source'].get('val_median', float('nan')):.3f}"
        + (f"，稳健 σ={m['threshold_source'].get('val_robust_sigma', float('nan')):.3f}"
           if m["threshold_source"].get("val_robust_sigma") is not None else ""),
        f"- 阈值：**{m['threshold']['threshold']:.6f}**",
        "",
        "| TP | TN | FP（误报） | FN（漏检） | Accuracy | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {m['threshold']['tp']} | {m['threshold']['tn']} | {m['threshold']['fp']} | "
        f"{m['threshold']['fn']} | {m['threshold']['accuracy']:.4f} | {m['threshold']['precision']:.4f} | "
        f"{m['threshold']['recall']:.4f} | {m['threshold']['f1']:.4f} |",
        "",
        "## 3. 分缺陷类型",
        "",
        "| 类型 | 数量 | 平均分数 | 最高分 | 检出数 | 检出率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t, v in m["per_type"].items():
        lines.append(f"| {t} | {v['count']} | {v['mean_score']:.4f} | {v['max_score']:.4f} | "
                     f"{v['detected_at_threshold']} | {v['detect_rate']:.2%} |")
    lines += [
        "",
        "## 4. 单张推理耗时",
        "",
        f"- 设备：{m['latency']['device']}",
        f"- 分辨率：{m['latency']['resolution'][1]}x{m['latency']['resolution'][0]}",
        f"- 是否含图片读取：{'是' if m['latency']['includes_image_read'] else '否'}",
        f"- 统计样本：{m['latency']['n_images']} 张 × {m['latency']['repeats']} 次",
        "",
        "| mean | p50 | p95 | p99 |",
        "|---:|---:|---:|---:|",
        f"| {m['latency']['mean_ms']:.1f} ms | {m['latency']['p50_ms']:.1f} ms | "
        f"{m['latency']['p95_ms']:.1f} ms | {m['latency']['p99_ms']:.1f} ms |",
        "",
        "## 5. 典型失败案例与边界样本",
        "",
        "| 情况 | 文件 | 分数 |",
        "|---|---|---:|",
    ]
    for f in m["failures"]["top_missed_abnormal"]:
        lines.append(f"| 漏检（异常判 OK） | {Path(f['file']).name} ({f['type']}) | {f['score']:.4f} |")
    for f in m["failures"]["top_false_alarm_normal"]:
        lines.append(f"| 正常样本最高分（阈值下未误报） | {Path(f['file']).name} ({f['type']}) | {f['score']:.4f} |")
    lines += [
        "",
        f"可视化：`results/{m['category']}/visualizations/comparison_grid.png`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from None
