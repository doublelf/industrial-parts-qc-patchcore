#!/usr/bin/env python
"""从真实 MVTec AD 中抽取极小子集，构造 smoke test 用的数据目录。

输出结构与官方一致：
    data/smoke/mvtec/<category>/train/good/*.png
    data/smoke/mvtec/<category>/test/<type>/*.png
    data/smoke/mvtec/<category>/ground_truth/<type>/*_mask.png
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

EXTS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构造 smoke test 小数据集")
    p.add_argument("--categories", nargs="+", default=["metal_nut"])
    p.add_argument("--src-root", default=None, help="真实数据根目录（默认取 config.yaml 的 data.root）")
    p.add_argument("--dst-root", default="data/smoke/mvtec")
    p.add_argument("--n-train", type=int, default=12)
    p.add_argument("--n-test-good", type=int, default=3)
    p.add_argument("--n-abnormal", type=int, default=2, help="每个缺陷类型抽取数量")
    return p.parse_args()


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS) if folder.is_dir() else []


def main() -> int:
    args = parse_args()
    cfg = load_config()
    log = setup_logger("make_smoke_data")
    src_root = resolve_path(args.src_root or cfg["data"]["root"])
    dst_root = resolve_path(args.dst_root)
    log.info("源数据: %s -> smoke 数据: %s", src_root, dst_root)

    for cat in args.categories:
        src = src_root / cat
        dst = dst_root / cat
        if not (src / "train" / "good").is_dir():
            log.error("源类别缺失: %s", src)
            return 1
        if dst.exists():
            shutil.rmtree(dst)

        train_imgs = _images(src / "train" / "good")[: args.n_train]
        for p in train_imgs:
            (dst / "train" / "good").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / "train" / "good" / p.name)

        good = _images(src / "test" / "good")[: args.n_test_good]
        for p in good:
            (dst / "test" / "good").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / "test" / "good" / p.name)

        copied_ab = 0
        for d in sorted((src / "test").iterdir()) if (src / "test").is_dir() else []:
            if not d.is_dir() or d.name == "good":
                continue
            imgs = _images(d)[: args.n_abnormal]
            if not imgs:
                continue
            (dst / "test" / d.name).mkdir(parents=True, exist_ok=True)
            (dst / "ground_truth" / d.name).mkdir(parents=True, exist_ok=True)
            gt_dir = src / "ground_truth" / d.name
            for p in imgs:
                shutil.copy2(p, dst / "test" / d.name / p.name)
                mask = gt_dir / f"{p.stem}_mask.png"
                if not mask.exists():
                    masks = _images(gt_dir)
                    mask = masks[0] if masks else None
                if mask is not None and mask.exists():
                    shutil.copy2(mask, dst / "ground_truth" / d.name / f"{p.stem}_mask.png")
                copied_ab += 1
        log.info("%s: train/good=%d | test/good=%d | 异常=%d", cat, len(train_imgs), len(good), copied_ab)
    log.info("smoke 数据就绪: %s", dst_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
