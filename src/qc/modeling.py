"""Anomalib PatchCore 的封装：数据模块、模型构建、预处理与打分。

设计说明：
- 训练阶段严格使用 Anomalib 的 ``MVTecAD`` datamodule（官方 train/test 划分）
  与 ``Patchcore`` 模型，由 Lightning ``Engine`` 完成记忆库构建；
- 推理/评估阶段直接调用 Patchcore 内部的 ``PatchcoreModel``（Anomalib 实现，
  就是论文里的最近邻 patch 匹配），拿到**未归一化**的原始异常分数与异常图，
  这样阈值物理意义清晰、便于跨类别比较；
- 预处理与 Anomalib ``Patchcore.configure_pre_processor`` 完全一致：
  Resize((256,256), antialias) + ImageNet Normalize。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# 数据模块与模型
# --------------------------------------------------------------------------- #
def build_datamodule(cfg: Dict, category: str):
    """构建 Anomalib MVTecAD datamodule（官方目录 + 官方 train/test 划分）。

    验证集从官方 train/good 中按 seed 确定性切出，且**不参与**记忆库构建，
    只用于确定判定阈值，避免任何 test 信息泄漏。
    """
    from anomalib.data import MVTecAD
    from anomalib.data.utils import ValSplitMode

    data_cfg = cfg["data"]
    val_cfg = data_cfg.get("val_split", {})
    val_enabled = bool(val_cfg.get("enabled", True))

    from .config import resolve_path

    dm = MVTecAD(
        root=str(resolve_path(data_cfg["root"])),
        category=category,
        train_batch_size=int(data_cfg["train_batch_size"]),
        eval_batch_size=int(data_cfg["eval_batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        test_split_mode="from_dir",  # 严格使用官方 test 目录
        val_split_mode=ValSplitMode.FROM_TRAIN if val_enabled else ValSplitMode.NONE,
        val_split_ratio=float(val_cfg.get("ratio", 0.2)),
        seed=int(cfg["project"]["seed"]),
    )
    return dm


def model_kwargs_from_config(cfg: Dict) -> Dict:
    m = cfg["model"]
    return {
        "backbone": m["backbone"],
        "layers": list(m["layers"]),
        "pre_trained": bool(m.get("pre_trained", True)),
        "coreset_sampling_ratio": float(m["coreset_sampling_ratio"]),
        "num_neighbors": int(m["num_neighbors"]),
    }


def build_model(cfg: Dict, pre_trained: bool | None = None):
    """构建 Anomalib Patchcore LightningModule。"""
    from anomalib.models import Patchcore

    kwargs = model_kwargs_from_config(cfg)
    if pre_trained is not None:
        kwargs["pre_trained"] = pre_trained
    # 关键：让 Anomalib 预处理器的 Resize 尺寸与 config 保持一致，
    # 否则训练按 256 建库、推理按其他尺寸打分会导致分数分布不一致。
    size = tuple(cfg["data"]["image_size"])
    kwargs["pre_processor"] = Patchcore.configure_pre_processor(image_size=(int(size[0]), int(size[1])))
    # 评估/推理阶段不需要内置 evaluator/visualizer，关闭以节省开销
    kwargs.update({"evaluator": False, "visualizer": False})
    return Patchcore(**kwargs)


def build_engine(cfg: Dict, default_root_dir: Path):
    """构建 Lightning Engine：单卡/CPU、无进度条、单 epoch（PatchCore 无需训练）。"""
    from anomalib.engine import Engine

    device = cfg["runtime"].get("resolved_device", "cpu")
    accelerator = "gpu" if str(device).startswith("cuda") else "cpu"
    return Engine(
        default_root_dir=str(default_root_dir),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator=accelerator,
        devices=1,
        max_epochs=1,
        deterministic=True,
    )


# --------------------------------------------------------------------------- #
# 预处理 / 打分
# --------------------------------------------------------------------------- #
def _build_transform(image_size: Sequence[int]):
    from torchvision.transforms import v2

    return v2.Compose([
        v2.Resize((int(image_size[0]), int(image_size[1])), antialias=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess(images: torch.Tensor, image_size: Sequence[int]) -> torch.Tensor:
    """images: (B,3,H,W) float in [0,1] -> 归一化后的 256x256 张量。"""
    return _build_transform(image_size)(images)


def load_image_tensor(path: str | Path) -> torch.Tensor:
    """读取单张图片为 (3,H,W) float [0,1] 张量（RGB）。"""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


@torch.no_grad()
def score_batch(model, images: torch.Tensor, image_size: Sequence[int], device: torch.device):
    """对一批原始图片张量打分，返回 (scores [B], anomaly_maps [B,1,H,W])。"""
    raw = model.model  # Anomalib PatchcoreModel
    raw.eval()
    x = preprocess(images, image_size).to(device)
    out = raw(x)
    return out.pred_score.detach().cpu().float(), out.anomaly_map.detach().cpu().float()


@torch.no_grad()
def score_paths(model, paths: Iterable[str | Path], image_size: Sequence[int], device: torch.device,
                batch_size: int = 8) -> Tuple[np.ndarray, List[np.ndarray]]:
    """逐文件读取并打分（用于 predict.py 的批量模式/耗时统计）。"""
    scores: List[float] = []
    maps: List[np.ndarray] = []
    batch: List[torch.Tensor] = []
    for p in paths:
        batch.append(load_image_tensor(p))
        if len(batch) == batch_size:
            s, m = score_batch(model, torch.stack(batch), image_size, device)
            scores.extend(s.tolist())
            maps.extend(m[:, 0].numpy())
            batch = []
    if batch:
        s, m = score_batch(model, torch.stack(batch), image_size, device)
        scores.extend(s.tolist())
        maps.extend(m[:, 0].numpy())
    return np.asarray(scores, dtype=np.float32), maps


# --------------------------------------------------------------------------- #
# 保存 / 加载
# --------------------------------------------------------------------------- #
def save_checkpoint(model, cfg: Dict, category: str, dest: Path, threshold_info: Dict | None = None) -> None:
    """保存模型权重（含记忆库 memory_bank）与复现所需的元信息。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "category": category,
        "model_kwargs": model_kwargs_from_config(cfg),
        "image_size": list(cfg["data"]["image_size"]),
        "seed": int(cfg["project"]["seed"]),
        "anomalib_version": _anomalib_version(),
        "threshold_info": threshold_info or {},
    }
    torch.save(payload, dest)


def load_checkpoint(cfg: Dict, category: str, ckpt_path: Path):
    """从 results/<category>/model.pt 恢复 Patchcore（pre_trained=False + 载入权重，离线可用）。"""
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"未找到模型文件: {ckpt_path}\n请先运行: python train.py --category {category}"
        )
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kwargs = payload.get("model_kwargs") or model_kwargs_from_config(cfg)
    kwargs = {k: v for k, v in kwargs.items() if k != "pre_trained"}
    from anomalib.models import Patchcore

    model = Patchcore(
        pre_trained=False,
        evaluator=False,
        visualizer=False,
        pre_processor=Patchcore.configure_pre_processor(
            image_size=(int(cfg["data"]["image_size"][0]), int(cfg["data"]["image_size"][1]))
        ),
        **kwargs,
    )
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"模型权重与当前 Anomalib 版本不匹配，多余键: {unexpected[:5]}")
    if model.model.memory_bank.numel() == 0:
        raise RuntimeError("加载的模型记忆库为空，请检查 model.pt 是否完整")
    model.eval()
    return model, payload


def _anomalib_version() -> str:
    try:
        import anomalib

        return anomalib.__version__
    except Exception:  # pragma: no cover
        return "unknown"
