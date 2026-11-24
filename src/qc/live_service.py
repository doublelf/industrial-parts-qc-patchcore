"""实时质检演示服务：模型常驻内存，逐张推理并记录统计。

职责边界：
- 只做“一张图推理一次”，不改变算法与阈值口径；
- 阈值读取 results/<category>/threshold.json（评估阶段在验证集上确定）；
- 预处理、打分完全复用 qc.modeling（与 predict.py 同一条路径）；
- 统计与 live_log.csv 由服务端维护，页面刷新后从服务端恢复，属于“接着累计”。
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .config import category_dir, load_config, load_run_config, resolve_path
from .device import device_description, resolve_device
from .logging_utils import setup_logger
from .modeling import load_checkpoint, score_batch
from .seed import set_seed
from .viz import save_prediction_overlay

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CSV_FIELDS = ["time", "category", "mode", "source", "file", "score", "decision", "latency_ms"]


@dataclass
class CategoryEntry:
    category: str
    model: Any
    cfg: Dict[str, Any]
    threshold: float
    threshold_info: Dict[str, Any]
    demo_files: List[Path]
    demo_source: str
    pointer: int = 0

    @property
    def image_size(self) -> tuple[int, int]:
        s = self.cfg["data"]["image_size"]
        return int(s[0]), int(s[1])


@dataclass
class Counters:
    total: int = 0
    ok: int = 0
    ng: int = 0
    latency_sum_ms: float = 0.0
    scores: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "ok": self.ok,
            "ng": self.ng,
            "defect_rate": (self.ng / self.total * 100.0) if self.total else 0.0,
            "avg_latency_ms": (self.latency_sum_ms / self.total) if self.total else 0.0,
            "scores": list(self.scores),
            "history": list(self.history)[:30],
        }

    def reset(self) -> None:
        self.total = self.ok = self.ng = 0
        self.latency_sum_ms = 0.0
        self.scores.clear()
        self.history.clear()


class LiveQCService:
    """常驻模型 + 推理 + 统计 + CSV 追加。"""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.cfg = load_config(config_path)
        self.live_dir = resolve_path(self.cfg["ui"]["live_dir"])
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.log = setup_logger("live", self.live_dir / "service.log")
        self.results_root = self.cfg["project"]["results_root"]
        self.device = resolve_device(self.cfg["runtime"].get("device", "auto"), self.log)
        self.cfg["runtime"]["resolved_device"] = str(self.device)
        set_seed(int(self.cfg["project"]["seed"]))
        self.lock = threading.RLock()
        self.counters = Counters()
        self.entries: Dict[str, CategoryEntry] = {}
        self.load_order: List[str] = []
        self.csv_path = self.live_dir / "live_log.csv"
        self._load_all()

    # ------------------------------------------------------------------ #
    # 启动加载
    # ------------------------------------------------------------------ #
    def _available_categories(self) -> List[str]:
        root = resolve_path(self.results_root)
        cats: List[str] = []
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if d.is_dir() and (d / "model.pt").exists() and (d / "threshold.json").exists():
                cats.append(d.name)
        return cats

    def _load_all(self) -> None:
        cats = self._available_categories()
        if not cats:
            raise RuntimeError("没有找到任何已训练模型（results/<category>/model.pt），请先运行 train.py")
        t0 = time.perf_counter()
        for cat in cats:
            out_dir = category_dir(self.results_root, cat)
            run_cfg = load_run_config(out_dir, fallback=self.cfg)
            run_cfg["runtime"]["resolved_device"] = str(self.device)
            model, _ = load_checkpoint(run_cfg, cat, out_dir / "model.pt")
            model.to(self.device)
            model.model.eval()
            thr_info = json.loads((out_dir / "threshold.json").read_text(encoding="utf-8"))
            files, source = self._resolve_demo_files(cat)
            self.entries[cat] = CategoryEntry(
                category=cat, model=model, cfg=run_cfg,
                threshold=float(thr_info["threshold"]), threshold_info=thr_info,
                demo_files=files, demo_source=source,
            )
            self.load_order.append(cat)
            self.log.info("已加载模型: %s | 阈值=%.4f | 设备=%s | 演示图片=%d 张（%s）",
                          cat, float(thr_info["threshold"]), device_description(self.device), len(files), source)
        self.log.info("全部模型加载完成: %d 类 | 耗时=%.1f s | 常驻内存",
                      len(self.entries), time.perf_counter() - t0)
        self.default_category = self.cfg["ui"].get("default_category") or cats[0]
        if self.default_category not in self.entries:
            self.default_category = cats[0]
        self._init_csv()

    def _resolve_demo_files(self, category: str) -> tuple[List[Path], str]:
        """优先 demo_images/<category>/，否则回退到 MVTec test 集。"""
        custom = resolve_path(self.cfg["ui"]["demo_images_root"]) / category
        if custom.is_dir():
            files = sorted(p for p in custom.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
            if files:
                return self._shuffle(files), f"demo_images/{category}"
        test_dir = resolve_path(self.cfg["data"]["root"]) / category / "test"
        files = sorted(p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS) if test_dir.is_dir() else []
        return self._shuffle(files), f"data/mvtec/{category}/test"

    def _shuffle(self, files: List[Path]) -> List[Path]:
        """固定 seed 打乱，让良品/不良品随机到达且每次启动顺序可复现。"""
        rng = random.Random(int(self.cfg["project"]["seed"]))
        files = list(files)
        rng.shuffle(files)
        return files

    # ------------------------------------------------------------------ #
    # CSV
    # ------------------------------------------------------------------ #
    def _init_csv(self) -> None:
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(CSV_FIELDS)

    def _append_csv(self, row: Dict[str, Any]) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row.get(k, "") for k in CSV_FIELDS])

    def _rotate_csv(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 1:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path.rename(self.live_dir / f"live_log_{stamp}.csv")
        self._init_csv()

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def _infer(self, entry: CategoryEntry, image_rgb: np.ndarray) -> tuple[float, np.ndarray, float]:
        """返回 (score, anomaly_map, latency_ms)。latency 与 predict.py 同口径：
        读图/解码 -> 预处理 -> 前向 -> 异常图，不含写叠加图。"""
        t0 = time.perf_counter()
        arr = (np.asarray(image_rgb, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
        scores, maps = score_batch(entry.model, x, entry.image_size, self.device)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return float(scores[0]), maps[0, 0].numpy(), latency_ms

    def _save_assets(self, entry: CategoryEntry, stem: str, image_rgb: np.ndarray,
                     anomaly_map: np.ndarray, score: float, decision: str) -> tuple[str, str]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:40]
        orig_name = f"{stamp}_{entry.category}_{safe}_orig.jpg"
        cv2.imwrite(str(self.live_dir / orig_name), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        overlay_path = save_prediction_overlay(
            self.live_dir, Path(f"{stamp}_{entry.category}_{safe}"),
            image_rgb, anomaly_map, score, decision, entry.threshold,
        )
        return f"/live/{orig_name}", f"/live/{overlay_path.name}"

    def _record(self, entry: CategoryEntry, source: str, file_name: str, mode: str,
                score: float, decision: str, latency_ms: float,
                image_url: str = "", overlay_url: str = "") -> Dict[str, Any]:
        row = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "category": entry.category,
            "mode": mode,
            "source": source,
            "file": file_name,
            "score": round(score, 6),
            "decision": decision,
            "latency_ms": round(latency_ms, 1),
        }
        self._append_csv(row)
        with self.lock:
            self.counters.total += 1
            if decision == "NG":
                self.counters.ng += 1
            else:
                self.counters.ok += 1
            self.counters.latency_sum_ms += latency_ms
            self.counters.scores.append(round(score, 4))
            self.counters.history.appendleft({
                **{k: row[k] for k in ("time", "category", "source", "file", "score", "decision", "latency_ms")},
                "image_url": image_url,
                "overlay_url": overlay_url,
            })
        return row

    def detect_bytes(self, data: bytes, category: str, filename: str, mode: str = "upload",
                     source: str = "upload") -> Dict[str, Any]:
        entry = self._entry(category)
        with Image.open(io.BytesIO(data)) as im:
            image_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        stem = Path(filename or "image.jpg").stem
        return self._run_and_package(entry, image_rgb, stem, f"{stem}.jpg", mode, source)

    def detect_path(self, path: Path, category: str, mode: str = "line") -> Dict[str, Any]:
        entry = self._entry(category)
        with Image.open(path) as im:
            image_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return self._run_and_package(entry, image_rgb, path.stem, path.name, mode, path.parent.name)

    def _run_and_package(self, entry: CategoryEntry, image_rgb: np.ndarray, stem: str,
                         file_name: str, mode: str, source: str) -> Dict[str, Any]:
        with self.lock:
            score, anomaly_map, latency_ms = self._infer(entry, image_rgb)
            decision = "NG" if score >= entry.threshold else "OK"
            image_url, overlay_url = self._save_assets(entry, stem, image_rgb, anomaly_map, score, decision)
            row = self._record(entry, source, file_name, mode, score, decision, latency_ms, image_url, overlay_url)
            counters = self.counters.as_dict()
        return {
            **{k: row[k] for k in ("time", "category", "mode", "source", "file", "score", "decision", "latency_ms")},
            "threshold": entry.threshold,
            "image_url": image_url,
            "overlay_url": overlay_url,
            "counters": counters,
        }

    def _entry(self, category: Optional[str]) -> CategoryEntry:
        cat = category or self.default_category
        if cat not in self.entries:
            raise KeyError(f"未建库的类别: {cat}")
        return self.entries[cat]

    # ------------------------------------------------------------------ #
    # 演示序列 / 状态 / 重置
    # ------------------------------------------------------------------ #
    def next_demo(self, category: Optional[str] = None) -> Dict[str, Any]:
        entry = self._entry(category)
        if not entry.demo_files:
            raise RuntimeError(f"{entry.category} 没有演示图片，请检查 data/mvtec 或 demo_images/{entry.category}")
        idx = entry.pointer % len(entry.demo_files)
        path = entry.demo_files[idx]
        looped = idx == 0 and entry.pointer > 0
        entry.pointer += 1
        result = self.detect_path(path, entry.category, mode="line")
        result["looped"] = looped
        result["position"] = idx + 1
        result["demo_total"] = len(entry.demo_files)
        return result

    def state(self, category: Optional[str] = None) -> Dict[str, Any]:
        entry = self._entry(category)
        return {
            "categories": list(self.entries.keys()),
            "default_category": entry.category,
            "threshold": entry.threshold,
            "threshold_strategy": entry.threshold_info.get("strategy"),
            "threshold_k": entry.threshold_info.get("k"),
            "threshold_n_val": entry.threshold_info.get("n_val"),
            "device": device_description(self.device),
            "image_size": list(entry.image_size),
            "demo_source": entry.demo_source,
            "demo_total": len(entry.demo_files),
            "default_interval_s": float(self.cfg["ui"].get("default_interval_s", 2.0)),
            "counters": self.counters.as_dict(),
            "notice": "演示模式：图片来自演示文件夹/测试集，不是真实产线对接，不良率不代表真实产线水平。",
        }

    def stats(self) -> Dict[str, Any]:
        return self.counters.as_dict()

    def reset(self) -> Dict[str, Any]:
        with self.lock:
            self.counters.reset()
            self._rotate_csv()
            for entry in self.entries.values():
                entry.pointer = 0
        self.log.info("统计已重置，历史 CSV 已归档")
        return self.counters.as_dict()
