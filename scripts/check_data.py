#!/usr/bin/env python
"""数据检查脚本：校验 MVTec AD 目录结构，统计各集合/缺陷类型图片数量。

用法：
    python scripts/check_data.py --category metal_nut
    python scripts/check_data.py --all
    python scripts/check_data.py --category metal_nut,screw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qc.config import load_config, resolve_path  # noqa: E402
from qc.data_check import (  # noqa: E402
    DOWNLOAD_HINT,
    format_markdown,
    list_categories,
    scan_category,
)
from qc.logging_utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVTec AD 数据结构检查")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 config.yaml）")
    parser.add_argument("--category", default=None, help="类别，可用逗号分隔，例如 metal_nut,screw")
    parser.add_argument("--all", action="store_true", help="检查 data/mvtec 下所有已存在类别")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = resolve_path(cfg["data"]["root"])
    log = setup_logger("check_data")

    log.info("=" * 70)
    log.info("MVTec AD 数据检查 | 数据根目录: %s", data_root)
    log.info("=" * 70)

    if args.all:
        categories = list_categories(data_root)
        if not categories:
            log.error("在 %s 下没有发现任何符合 MVTec 结构的类别目录", data_root)
            log.error("\n%s", DOWNLOAD_HINT)
            return 1
    elif args.category:
        categories = [c.strip() for c in args.category.split(",") if c.strip()]
    else:
        categories = list(cfg.get("categories", {}).keys()) or ["metal_nut"]
        log.info("未指定 --category，使用配置中的类别: %s", categories)

    if not data_root.is_dir():
        log.error("数据根目录不存在: %s", data_root)
        log.error("\n%s", DOWNLOAD_HINT)
        return 1

    all_stats = [scan_category(data_root, c) for c in categories]
    for st in all_stats:
        log.info("-" * 70)
        log.info("类别: %s", st.category)
        log.info("  路径: %s", st.root)
        log.info("  train/good       : %d", st.train_good)
        for dtype, n in st.test_counts.items():
            log.info("  test/%-14s: %d", dtype, n)
        for dtype, n in st.gt_counts.items():
            log.info("  ground_truth/%-4s: %d", dtype, n)
        log.info("  合计: 正常训练 %d | 测试异常 %d | 测试正常 %d | 掩码 %d",
                 st.train_good, st.test_abnormal, st.test_counts.get("good", 0), st.gt_total)
        for w in st.warnings:
            log.warning("  [告警] %s", w)
        if not st.warnings:
            log.info("  结构检查: 通过")

    md = format_markdown(all_stats)
    out_path = resolve_path(cfg["project"]["results_root"]) / "data_check.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    log.info("统计表已保存: %s", out_path)

    # 结构性问题：类别目录缺失 / train 为空 / test 为空 -> 非 0 退出码，方便 CI 拦截
    fatal = [st for st in all_stats if (not st.root.is_dir()) or st.train_good == 0 or not st.test_counts]
    if fatal:
        log.error("存在数据缺失的类别: %s", [st.category for st in fatal])
        log.error("\n%s", DOWNLOAD_HINT)
        return 1
    log.info("全部类别结构检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
