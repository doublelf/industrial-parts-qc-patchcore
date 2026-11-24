#!/usr/bin/env python
"""汇总多个类别的评估结果，生成 results/summary.md。

用法：
    python scripts/summarize.py                # 默认汇总 config.yaml 中启用的类别
    python scripts/summarize.py --categories metal_nut screw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qc.config import load_config, resolve_path  # noqa: E402
from qc.logging_utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="汇总多类别结果")
    p.add_argument("--config", default=None)
    p.add_argument("--categories", nargs="+", default=None)
    return p.parse_args()


def _fmt(v, nd=4, suffix=""):
    if v is None:
        return "不支持"
    return f"{v:.{nd}f}{suffix}"


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    root = resolve_path(cfg["project"]["results_root"])
    cats = args.categories or list(cfg.get("categories", {}).keys())
    log = setup_logger("summarize")

    loaded = []
    for c in cats:
        p = root / c / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            # 若有独立耗时基准（更多重复次数、更稳定），优先使用它
            bench = root / c / "latency_bench.json"
            if bench.exists():
                b = json.loads(bench.read_text(encoding="utf-8"))
                m["latency"] = {**b["end_to_end"], "source": "latency_bench.json",
                                "n_images": b["n_images"], "repeats": b["repeats"]}
            else:
                m["latency"]["source"] = "evaluate.py"
            loaded.append(m)
        else:
            log.warning("缺少 %s，跳过（先运行 train.py + evaluate.py）", p)
    if not loaded:
        log.error("没有任何可汇总的 metrics.json")
        return 1

    lines = [
        "# PatchCore 多类别汇总对比",
        "",
        f"- 数据集：MVTec AD（CC BY-NC-SA 4.0，仅研发验证）",
        f"- 随机种子：{cfg['project']['seed']}",
        f"- backbone：{cfg['model']['backbone']} | layers：{cfg['model']['layers']} | "
        f"coreset 比例：{cfg['model']['coreset_sampling_ratio']} | 输入：{cfg['data']['image_size'][0]}x{cfg['data']['image_size'][1]}",
        f"- 阈值策略：{cfg['threshold']['strategy']}（验证集正常样本 median + k×1.4826×MAD，k={cfg['threshold'].get('k', 3.0)}；不使用 test）",
        "",
        "## 1. 指标与耗时对比",
        "",
        "| 类别 | 图像 AUROC | 图像 AP | 像素 AUROC | 像素 AP | AU-PRO | 阈值 | FP | FN | 单张 mean | 单张 p99 | 设备 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in loaded:
        lat = m.get("latency", {})
        aupro = _fmt(m.get("aupro"))
        lines.append(
            f"| {m['category']} | {_fmt(m['image_auroc'])} | {_fmt(m['image_ap'])} | "
            f"{_fmt(m['pixel_auroc'])} | {_fmt(m['pixel_ap'])} | {aupro} | "
            f"{m['threshold']['threshold']:.4f} | {m['threshold']['fp']} | {m['threshold']['fn']} | "
            f"{lat.get('mean_ms', 0):.1f} ms | {lat.get('p99_ms', 0):.1f} ms | {m.get('device', '-')} |"
        )

    lines += ["", "## 2. 典型失败案例与边界样本（每类 3 个）", "",
              "| 类别 | 情况 | 文件 | 分数 | 说明 |", "|---|---|---|---:|---|"]
    for m in loaded:
        fails = m.get("failures", {})
        for f in fails.get("top_missed_abnormal", [])[:3]:
            lines.append(f"| {m['category']} | 漏检 | {Path(f['file']).name} | {f['score']:.4f} | "
                         f"{f['type']} 缺陷分数低于阈值 |")
        for f in fails.get("top_false_alarm_normal", [])[:3]:
            lines.append(f"| {m['category']} | 正常最高分 | {Path(f['file']).name} | {f['score']:.4f} | "
                         f"正常样本中分数最高，未超过阈值 |")

    lines += ["", "## 3. 复现信息", "",
              "| 类别 | seed | Anomalib | 测试集数量 | 阈值来源样本数 |",
              "|---|---:|---|---:|---:|"]
    for m in loaded:
        lines.append(f"| {m['category']} | {m['seed']} | {m['anomalib_version']} | {m['n_test']} | "
                     f"{m['threshold_source']['n_val']} |")
    lines += ["", "> 所有指标均来自 `results/<category>/metrics.json` 的实际运行输出；",
              "> 耗时优先取自 `results/<category>/latency_bench.json`（20 张 × 5 次端到端，含读图），",
              "> 没有基准文件时回退到 `evaluate.py` 的测量值。CPU 共享开发机耗时存在 ±30% 波动。", ""]

    out = root / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("汇总已保存: %s", out)
    for m in loaded:
        log.info("%s: AUROC=%.4f AP=%.4f pixel_AUROC=%.4f FP=%d FN=%d mean=%.1fms p99=%.1fms",
                 m["category"], m["image_auroc"], m["image_ap"], m["pixel_auroc"],
                 m["threshold"]["fp"], m["threshold"]["fn"], m["latency"]["mean_ms"], m["latency"]["p99_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
