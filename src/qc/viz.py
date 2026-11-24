"""可视化：热力图叠加、单图输出、原图/GT/热力图三列对比图。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np


def normalize_map(anomaly_map: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """异常图 min-max 归一化到 [0,1]。"""
    m = np.asarray(anomaly_map, dtype=np.float32)
    lo = float(m.min()) if vmin is None else float(vmin)
    hi = float(m.max()) if vmax is None else float(vmax)
    return np.clip((m - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def heatmap_overlay(image_rgb: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45,
                    vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """把异常图（H,W）叠加到原图（H,W,3 uint8）上，返回 uint8 RGB 叠加图。"""
    h, w = image_rgb.shape[:2]
    m = cv2.resize(np.asarray(anomaly_map, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m = normalize_map(m, vmin, vmax)
    heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (image_rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def put_label(image_rgb: np.ndarray, text: str) -> np.ndarray:
    """在左上角写一行文字，便于人工核查。"""
    out = image_rgb.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 470), 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def save_prediction_overlay(out_dir: Path, source_path: Path, image_rgb: np.ndarray,
                            anomaly_map: np.ndarray, score: float, decision: str,
                            threshold: float) -> Path:
    """保存单图叠加图：<原图名>_score<分数>_<OK|NG>.png"""
    overlay = heatmap_overlay(image_rgb, anomaly_map)
    overlay = put_label(overlay, f"{decision}  score={score:.4f}  thr={threshold:.4f}")
    name = f"{source_path.stem}_score{score:.4f}_{decision}.png"
    out_path = out_dir / name
    save_image(out_path, overlay)
    return out_path


def make_comparison_grid(samples: Sequence[Dict], out_path: Path, title: str = "PatchCore inspection") -> None:
    """三列对比图：原图 | GT mask | 热力图叠加。

    samples: [{"image": HxWx3 uint8, "mask": HxW uint8 or None, "map": HxW,
               "score": float, "decision": str, "name": str, "label": int}]
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.6 * n + 0.6), squeeze=False)
    for i, s in enumerate(samples):
        img = s["image"]
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(s["name"], fontsize=7)
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=9, pad=6)
        if s.get("mask") is not None:
            axes[i, 1].imshow(s["mask"], cmap="gray")
        else:
            axes[i, 1].text(0.5, 0.5, "no GT (normal)", ha="center", va="center", fontsize=8)
        if i == 0:
            axes[i, 1].set_title("Ground truth", fontsize=9, pad=6)
        axes[i, 2].imshow(heatmap_overlay(img, s["map"]))
        if i == 0:
            axes[i, 2].set_title("PatchCore overlay", fontsize=9, pad=6)
        for j in range(3):
            axes[i, j].axis("off")
        axes[i, 2].text(0.99, 0.02, f"{s['score']:.4f} {s['decision']}", transform=axes[i, 2].transAxes,
                        ha="right", va="bottom", fontsize=7, color="white",
                        bbox={"facecolor": "black", "alpha": 0.5, "pad": 1.5})
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
