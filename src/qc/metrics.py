"""评估指标：图像级/像素级 AUROC、AP，AU-PRO，阈值选择与耗时统计。

指标全部用测试集原始输出计算，不做任何针对 test 的阈值调优：
- 阈值来自验证集（官方 train/good 中切出的正常样本）分数分位数；
- AU-PRO 调用 Anomalib 2.7.1 自带的 ``anomalib.metrics.AUPRO``。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------- #
# 图像级 / 像素级指标
# --------------------------------------------------------------------------- #
def image_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def image_ap(labels: Sequence[int], scores: Sequence[float]) -> float:
    return float(average_precision_score(np.asarray(labels), np.asarray(scores)))


def pixel_auroc(pixel_labels: np.ndarray, pixel_scores: np.ndarray) -> float:
    return float(roc_auc_score(pixel_labels, pixel_scores))


def pixel_ap(pixel_labels: np.ndarray, pixel_scores: np.ndarray) -> float:
    return float(average_precision_score(pixel_labels, pixel_scores))


def compute_aupro(maps: Sequence[np.ndarray], masks: Sequence[np.ndarray], fpr_limit: float = 0.3) -> float | None:
    """调用 Anomalib 的 AUPRO 计算区域重叠曲线下面积；不可用时返回 None。

    maps: 每张图的异常图（H,W），会自动做全局 min-max 归一化到 [0,1]；
    masks: 对应 GT 掩码（H,W），取值 {0,1}，尺寸需与异常图一致。
    """
    try:
        import torch
        # anomalib.metrics.AUPRO 是要求 Batch 字段的包装类，这里直接用底层 torchmetrics 实现
        from anomalib.metrics.aupro import _AUPRO
    except Exception:
        return None

    if len(maps) == 0:
        return None
    all_vals = np.concatenate([m.reshape(-1) for m in maps])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    scale = max(vmax - vmin, 1e-12)

    preds = np.stack([(m - vmin) / scale for m in maps]).astype(np.float32)
    targets = np.stack([(m > 0.5).astype(np.float32) for m in masks])
    if targets.sum() == 0:
        return None

    metric = _AUPRO(fpr_limit=fpr_limit)
    metric.update(torch.from_numpy(preds), torch.from_numpy(targets))
    value = metric.compute()
    return float(value.item() if hasattr(value, "item") else value)


# --------------------------------------------------------------------------- #
# 阈值选择（仅允许使用验证集正常样本）
# --------------------------------------------------------------------------- #
def select_threshold(val_scores: Sequence[float], strategy: str = "val_normal_robust",
                     quantile: float = 0.99, k: float = 3.0) -> Dict[str, float]:
    """在验证集正常样本分数分布上确定判定阈值（绝不使用 test）。

    策略：
    - ``val_normal_robust``：median + k * 1.4826*MAD（稳健 3σ，抗离群正常样本）
    - ``val_normal_quantile``：验证集正常样本分数的 quantile 分位
    - ``val_normal_max``：验证集正常样本分数最大值
    - ``val_normal_mean_std``：mean + k*std（非稳健，作为对照）

    返回 dict: threshold, strategy, quantile/k(可选), val_fp_rate, n_val
    """
    arr = np.asarray(val_scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("验证集分数为空，无法确定阈值")
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust_sigma = 1.4826 * mad
    if strategy == "val_normal_max":
        thr = float(arr.max())
    elif strategy == "val_normal_quantile":
        thr = float(np.quantile(arr, quantile))
    elif strategy == "val_normal_mean_std":
        thr = float(arr.mean() + k * arr.std())
    elif strategy == "val_normal_robust":
        thr = float(median + k * robust_sigma)
    else:
        raise ValueError(f"未知阈值策略: {strategy}")
    val_fp_rate = float((arr >= thr).mean())
    return {
        "threshold": thr,
        "strategy": strategy,
        "quantile": float(quantile) if strategy == "val_normal_quantile" else None,
        "k": float(k) if strategy in ("val_normal_robust", "val_normal_mean_std") else None,
        "val_fp_rate": val_fp_rate,
        "n_val": int(arr.size),
        "val_median": median,
        "val_mad": mad,
        "val_robust_sigma": robust_sigma,
        "val_mean": float(arr.mean()),
        "val_std": float(arr.std()),
        "val_max": float(arr.max()),
        "val_min": float(arr.min()),
    }


def threshold_confusion(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    """按固定阈值统计图像级混淆矩阵：FP=正常判NG，FN=异常判OK。"""
    labels_arr = np.asarray(labels).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(((pred == 1) & (labels_arr == 1)).sum())
    tn = int(((pred == 0) & (labels_arr == 0)).sum())
    fp = int(((pred == 1) & (labels_arr == 0)).sum())
    fn = int(((pred == 0) & (labels_arr == 1)).sum())
    total = int(labels_arr.size)
    return {
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "n_normal": int((labels_arr == 0).sum()),
        "n_abnormal": int((labels_arr == 1).sum()),
    }


# --------------------------------------------------------------------------- #
# 耗时统计
# --------------------------------------------------------------------------- #
def latency_stats(times_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "n_samples": int(arr.size),
    }


def image_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    return {
        "image_auroc": image_auroc(labels, scores),
        "image_ap": image_ap(labels, scores),
    }


def pixel_metrics(pixel_labels: np.ndarray, pixel_scores: np.ndarray) -> Dict[str, float]:
    return {
        "pixel_auroc": pixel_auroc(pixel_labels, pixel_scores),
        "pixel_ap": pixel_ap(pixel_labels, pixel_scores),
    }


def summarize_per_type(labels: Sequence[int], scores: Sequence[float], types: Sequence[str],
                       threshold: float) -> Dict[str, Dict[str, float]]:
    """按缺陷类型统计平均分数与检出率，便于定位失败案例。"""
    out: Dict[str, Dict[str, float]] = {}
    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)
    for t in sorted(set(types)):
        mask = np.asarray([x == t for x in types])
        if mask.sum() == 0:
            continue
        s = scores_arr[mask]
        y = labels_arr[mask]
        out[t] = {
            "count": int(mask.sum()),
            "mean_score": float(s.mean()),
            "max_score": float(s.max()),
            "min_score": float(s.min()),
            "detected_at_threshold": int((s >= threshold).sum()),
            "detect_rate": float((s >= threshold).mean()),
            "label": int(y[0]) if len(y) else 0,
        }
    return out
