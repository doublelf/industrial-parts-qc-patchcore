"""项目级配置加载与路径工具。

所有脚本共享同一个 config.yaml，禁止在业务代码里硬编码类别/尺寸/阈值。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# src/qc/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """读取 YAML 配置并做基础校验。"""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误（应为映射）: {cfg_path}")
    cfg["_config_path"] = str(cfg_path)
    _validate(cfg)
    return cfg


def _validate(cfg: Dict[str, Any]) -> None:
    for key in ("project", "data", "model", "threshold", "runtime", "evaluation"):
        if key not in cfg:
            raise KeyError(f"config.yaml 缺少必需段落: {key}")
    size = cfg["data"]["image_size"]
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        raise ValueError(f"data.image_size 必须是 [H, W]，当前为 {size}")


def resolve_path(path_like: str | Path) -> Path:
    """相对路径一律相对项目根目录解析。"""
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def category_dir(results_root: str | Path, category: str) -> Path:
    """results/<category>/"""
    return resolve_path(results_root) / category


def outputs_dir(outputs_root: str | Path, category: str) -> Path:
    """outputs/<category>/"""
    return resolve_path(outputs_root) / category


def snapshot_config(cfg: Dict[str, Any], dest: Path) -> None:
    """把本次运行生效的配置（含命令行覆盖项）落盘，便于复现实验。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = copy.deepcopy(cfg)
    data.pop("_config_path", None)
    with dest.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """按点分路径覆盖配置，例如 {"model.coreset_sampling_ratio": 0.05}。"""
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def load_run_config(out_dir: Path, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """优先读取 results/<category>/config.yaml（训练时落盘的快照），保证评估可复现。"""
    snapshot = Path(out_dir) / "config.yaml"
    if snapshot.exists():
        return load_config(snapshot)
    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"未找到运行配置快照: {snapshot}，请先运行 train.py")

