"""设备选择：优先 GPU，不可用时自动回退 CPU 并在日志说明。

随机种子统一由 ``qc.seed.set_seed`` 负责（此处仅重导出，兼容旧导入路径）。
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .seed import set_seed  # noqa: F401  (re-export)


def resolve_device(preference: str = "auto", logger: Optional[logging.Logger] = None) -> "torch.device":
    """按 auto/cuda/cpu 解析运行设备，返回 torch.device 并打印说明。"""
    import torch

    log = logger or logging.getLogger("qc")
    preference = (preference or "auto").lower()
    cuda_ok = torch.cuda.is_available()

    log.info("torch=%s | cuda_available=%s | cuda_version=%s", torch.__version__, cuda_ok, torch.version.cuda)
    if cuda_ok:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log.info("检测到 GPU: %s (%.1f GiB)", gpu_name, total_gb)
        except Exception as exc:  # pragma: no cover
            log.warning("读取 GPU 信息失败: %s", exc)

    if preference == "cpu":
        log.info("按配置强制使用 CPU。")
        return torch.device("cpu")
    if preference == "cuda":
        if cuda_ok:
            return torch.device("cuda")
        log.warning("配置要求 CUDA，但当前环境不可用（无 GPU 或 torch 为 CPU 版），自动回退到 CPU。")
        return torch.device("cpu")
    if cuda_ok:
        log.info("设备策略 auto -> 使用 GPU (cuda:0)。")
        return torch.device("cuda")
    log.warning("设备策略 auto -> 未检测到可用 CUDA，自动回退到 CPU（推理/建库会明显变慢）。")
    return torch.device("cpu")


def device_description(device: "torch.device") -> str:
    """给报告用的设备描述字符串。"""
    import torch

    if device.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return f"CPU: {os.cpu_count()} threads"
