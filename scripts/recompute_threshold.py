#!/usr/bin/env python
"""用已保存的验证集分数重新计算判定阈值（仅验证集，不接触 test）。

适用场景：训练已完成、只想调整阈值策略（例如从 q99 换成稳健 3σ）时，
无需重新建库（建库在 CPU 上可能需要几十分钟）。

用法：
    python scripts/recompute_threshold.py --categories metal_nut screw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qc.config import category_dir, load_config, load_run_config  # noqa: E402
from qc.logging_utils import setup_logger  # noqa: E402
from qc.metrics import select_threshold  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="重算验证集阈值")
    p.add_argument("--config", default=None)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--strategy", default=None)
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--quantile", type=float, default=None)
    args = p.parse_args()

    base_cfg = load_config(args.config)
    log = setup_logger("recompute_threshold")
    for cat in args.categories:
        out_dir = category_dir(base_cfg["project"]["results_root"], cat)
        run_info_path = out_dir / "run_info.json"
        if not run_info_path.exists():
            log.error("%s: 缺少 run_info.json，请先训练", cat)
            return 1
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        cfg = load_run_config(out_dir, fallback=base_cfg)
        # 策略以 config.yaml（命令行 --config）为准；快照可能来自旧策略
        strategy = args.strategy or base_cfg["threshold"]["strategy"]
        k = args.k if args.k is not None else float(base_cfg["threshold"].get("k", 3.0))
        quantile = args.quantile if args.quantile is not None else float(base_cfg["threshold"].get("quantile", 0.99))

        info = select_threshold(run_info["val_scores"], strategy, quantile, k)
        old = json.loads((out_dir / "threshold.json").read_text(encoding="utf-8")) if (out_dir / "threshold.json").exists() else {}
        (out_dir / "threshold.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        run_info["threshold"] = info
        run_info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
        # 同步配置快照，保证评估快照与阈值策略一致
        cfg["threshold"]["strategy"] = strategy
        cfg["threshold"]["k"] = k
        cfg["threshold"]["quantile"] = quantile
        import yaml

        (out_dir / "config.yaml").write_text(
            yaml.safe_dump({k2: v for k2, v in cfg.items() if k2 != "_config_path"},
                           allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        log.info("%s: %s -> 新阈值 %.4f | k=%s | 验证集 %d 张 | 验证集误报率 %.4f",
                 cat, old.get("strategy", "n/a"), info["threshold"], info["k"], info["n_val"], info["val_fp_rate"])
    log.info("完成。下一步: python evaluate.py --category <category>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
