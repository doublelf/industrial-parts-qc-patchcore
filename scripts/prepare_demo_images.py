#!/usr/bin/env python
"""从 MVTec AD test 集复制一批图片到 demo_images/<category>/，作为固定演示素材。

用法：
    python scripts/prepare_demo_images.py --category metal_nut
    python scripts/prepare_demo_images.py --category screw --n-good 15 --n-defect 5
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qc.config import load_config, resolve_path  # noqa: E402
from qc.logging_utils import setup_logger  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> int:
    p = argparse.ArgumentParser(description="准备 demo_images 演示素材")
    p.add_argument("--config", default=None)
    p.add_argument("--category", required=True)
    p.add_argument("--n-good", type=int, default=12)
    p.add_argument("--n-defect", type=int, default=4, help="每个缺陷类型复制的数量")
    p.add_argument("--force", action="store_true", help="清空已有 demo_images/<category> 后重新复制")
    args = p.parse_args()

    cfg = load_config(args.config)
    log = setup_logger("prepare_demo_images")
    src_test = resolve_path(cfg["data"]["root"]) / args.category / "test"
    dst = resolve_path(cfg["ui"]["demo_images_root"]) / args.category
    if not src_test.is_dir():
        log.error("找不到 MVTec test 目录: %s", src_test)
        return 1
    if dst.exists() and args.force:
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    def copy(src_dir: Path, dst_dir: Path, limit: int) -> int:
        files = sorted(q for q in src_dir.iterdir() if q.is_file() and q.suffix.lower() in EXTS)[:limit]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dst_dir / f.name)
        return len(files)

    n_good = copy(src_test / "good", dst / "good", args.n_good)
    n_ab = 0
    for d in sorted(src_test.iterdir()):
        if d.is_dir() and d.name != "good":
            n_ab += copy(d, dst / d.name, args.n_defect)
    log.info("%s: 复制正常 %d 张、异常 %d 张 -> %s", args.category, n_good, n_ab, dst)
    log.info("重启演示服务（python app.py）后生效；删除该目录可恢复使用完整 test 集")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
