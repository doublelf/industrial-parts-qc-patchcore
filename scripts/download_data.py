#!/usr/bin/env python
"""从 Hugging Face 镜像下载 MVTec AD 指定类别（保留官方目录结构）。

说明：MVTec AD 官方站需要填表，本脚本使用社区镜像 `foersben/mvtec-ad`，
该仓库按官方目录结构组织数据。数据许可仍为 CC BY-NC-SA 4.0，仅限非商业研发。

用法：
    python scripts/download_data.py --categories metal_nut screw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qc.config import load_config, resolve_path  # noqa: E402
from qc.logging_utils import setup_logger  # noqa: E402

REPO_ID = "foersben/mvtec-ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 MVTec AD 指定类别")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--categories", nargs="+", default=["metal_nut", "screw"], help="要下载的类别")
    parser.add_argument("--repo-id", default=REPO_ID, help="Hugging Face 数据集仓库 id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logger("download_data", PROJECT_ROOT / "logs" / "download_data.log")
    cfg = load_config(args.config)
    dest = resolve_path(cfg["data"]["root"])
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("缺少 huggingface_hub，请先执行: pip install huggingface_hub")
        return 1

    patterns = [f"{c}/*" for c in args.categories]
    log.info("从 %s 下载类别: %s", args.repo_id, args.categories)
    log.info("目标目录: %s", dest)

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=str(dest),
        )
    except Exception as exc:  # 网络/仓库不可达时给出手动下载提示
        log.error("自动下载失败: %s: %s", type(exc).__name__, exc)
        log.error("请按 data/README_download.md 手动下载后解压到 %s", dest)
        return 1

    log.info("下载完成: %s", path)
    for c in args.categories:
        n_train = len(list((dest / c / "train" / "good").glob("*"))) if (dest / c / "train" / "good").is_dir() else 0
        n_test = len(list((dest / c / "test").rglob("*.png"))) if (dest / c / "test").is_dir() else 0
        log.info("  %s: train/good=%d, test 图片=%d", c, n_train, n_test)
    log.info("下一步: python scripts/check_data.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
