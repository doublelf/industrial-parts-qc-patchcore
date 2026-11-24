"""MVTec AD 目录检查与统计。

官方目录结构（Anomalib MVTecAD datamodule 默认结构）：
    data/mvtec/<category>/train/good/*.png
    data/mvtec/<category>/test/good/*.png
    data/mvtec/<category>/test/<defect_type>/*.png
    data/mvtec/<category>/ground_truth/<defect_type>/*_mask.png

本模块只做只读扫描：训练/评估前调用，数量为 0 或目录缺失时给出明确告警，
并复用与 data/README_download.md 相同的下载提示，而不是直接抛异常崩溃。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import resolve_path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DOWNLOAD_HINT = """\
[数据缺失] 未找到可用的 MVTec AD 数据。请按以下步骤准备（详见 data/README_download.md）：
  1) 官网 https://www.mvtec.com/company/research/datasets/mvtec-ad 填写表格后下载 mvtec_anomaly_detection.tar.xz；
     或使用官方镜像仓库（仅研发用途）：
       huggingface-cli download foersben/mvtec-ad --repo-type dataset --include "metal_nut/*" "screw/*" --local-dir data/mvtec
  2) 解压后确保结构为：
       data/mvtec/metal_nut/train/good/*.png
       data/mvtec/metal_nut/test/good/*.png
       data/mvtec/metal_nut/test/<defect_type>/*.png
       data/mvtec/metal_nut/ground_truth/<defect_type>/*_mask.png
     （screw 类别同理；目录名大小写敏感）
  3) 重新运行： python scripts/check_data.py --category metal_nut
许可提醒：MVTec AD 为 CC BY-NC-SA 4.0，仅限非商业研发/验证使用。"""


@dataclass
class CategoryStats:
    category: str
    root: Path
    train_good: int = 0
    test_counts: Dict[str, int] = field(default_factory=dict)
    gt_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def test_abnormal(self) -> int:
        return sum(v for k, v in self.test_counts.items() if k != "good")

    @property
    def gt_total(self) -> int:
        return sum(self.gt_counts.values())


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _subdirs(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def scan_category(root: str | Path, category: str) -> CategoryStats:
    """扫描单个类别的官方目录并统计各集合图片数量。"""
    root_path = resolve_path(root)
    cat_root = root_path / category
    stats = CategoryStats(category=category, root=cat_root)

    if not cat_root.is_dir():
        stats.warnings.append(f"类别目录不存在: {cat_root}")
        return stats

    stats.train_good = _count_images(cat_root / "train" / "good")
    if stats.train_good == 0:
        stats.warnings.append("train/good 图片数量为 0，无法建模正常样本")

    test_types = _subdirs(cat_root / "test")
    if not test_types:
        stats.warnings.append("test/ 下没有任何缺陷类型子目录")
    for dtype in test_types:
        n = _count_images(cat_root / "test" / dtype)
        stats.test_counts[dtype] = n
        if n == 0:
            stats.warnings.append(f"test/{dtype} 图片数量为 0")

    gt_types = _subdirs(cat_root / "ground_truth")
    for dtype in gt_types:
        n = _count_images(cat_root / "ground_truth" / dtype)
        stats.gt_counts[dtype] = n
        if n == 0:
            stats.warnings.append(f"ground_truth/{dtype} 掩码数量为 0")

    # 缺陷类型必须在 test 与 ground_truth 中成对出现，否则像素级指标会缺样本
    for dtype in test_types:
        if dtype == "good":
            continue
        if dtype not in gt_types:
            stats.warnings.append(f"test/{dtype} 存在但缺少 ground_truth/{dtype} 掩码目录")
        elif stats.gt_counts.get(dtype, 0) != stats.test_counts.get(dtype, 0):
            stats.warnings.append(
                f"test/{dtype} ({stats.test_counts.get(dtype, 0)}) 与 "
                f"ground_truth/{dtype} ({stats.gt_counts.get(dtype, 0)}) 数量不一致"
            )
    return stats


def format_markdown(stats_all: List[CategoryStats]) -> str:
    """生成人类可读的统计表（Markdown）。"""
    lines = ["# MVTec AD 数据检查统计", "", "| 类别 | 集合 | 类型 | 数量 | 状态 |", "|---|---|---|---:|---|"]
    for st in stats_all:
        if not st.root.is_dir():
            lines.append(f"| {st.category} | - | - | 0 | 类别目录缺失 |")
            continue
        lines.append(f"| {st.category} | train | good | {st.train_good} | {'OK' if st.train_good else '报警: 为 0'} |")
        for dtype, n in st.test_counts.items():
            lines.append(f"| {st.category} | test | {dtype} | {n} | {'OK' if n else '报警: 为 0'} |")
        for dtype, n in st.gt_counts.items():
            lines.append(f"| {st.category} | ground_truth | {dtype} | {n} | {'OK' if n else '报警: 为 0'} |")
    lines.append("")
    for st in stats_all:
        if st.warnings:
            lines.append(f"## {st.category} 告警")
            for w in st.warnings:
                lines.append(f"- {w}")
            lines.append("")
    return "\n".join(lines)


def require_category(root: str | Path, category: str) -> CategoryStats:
    """训练/评估前调用：结构不完整时抛出带下载提示的异常。"""
    st = scan_category(root, category)
    problematic = [w for w in st.warnings if "数量为 0" in w or "不存在" in w]
    if not st.root.is_dir():
        raise FileNotFoundError(f"{DOWNLOAD_HINT}\n（缺失路径: {st.root}）")
    if st.train_good == 0:
        raise FileNotFoundError(f"{DOWNLOAD_HINT}\n（{st.root} 下 train/good 为空）")
    if not st.test_counts:
        raise FileNotFoundError(f"{DOWNLOAD_HINT}\n（{st.root} 下 test/ 为空）")
    if problematic:
        # 仍允许继续，但把问题打印出来（例如缺少某类掩码只影响像素指标）
        st.warnings.extend(problematic)
    return st


def data_present(root: str | Path, category: str) -> bool:
    p = resolve_path(root) / category / "train" / "good"
    return p.is_dir() and _count_images(p) > 0


def list_categories(root: str | Path) -> List[str]:
    """列出 data/mvtec 下已存在的类别（用于自动发现）。"""
    root_path = resolve_path(root)
    if not root_path.is_dir():
        return []
    return sorted(
        p.name
        for p in root_path.iterdir()
        if p.is_dir() and (p / "train" / "good").is_dir() and (p / "test").is_dir()
    )
