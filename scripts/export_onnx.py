#!/usr/bin/env python
"""尝试用 Anomalib 的 Engine.export 把训练好的 PatchCore 导出为 ONNX，并做数值验证。

用法：
    python scripts/export_onnx.py --category metal_nut
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qc.config import category_dir, load_config, load_run_config  # noqa: E402
from qc.logging_utils import setup_logger  # noqa: E402
from qc.modeling import load_checkpoint, load_image_tensor, preprocess, score_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchCore ONNX 导出与验证")
    p.add_argument("--config", default=None)
    p.add_argument("--category", required=True)
    p.add_argument("--results-root", default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--sample", default=None, help="用于数值验证的图片路径（默认取测试集第一张）")
    return p.parse_args()


def find_sample(out_dir: Path, category: str, data_root: Path) -> Path | None:
    test_dir = data_root / category / "test"
    if test_dir.is_dir():
        for p in sorted(test_dir.rglob("*.png")):
            return p
    return None


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    results_root = args.results_root or cfg["project"]["results_root"]
    out_dir = category_dir(results_root, args.category)
    run_cfg = load_run_config(out_dir, fallback=cfg)
    log = setup_logger("export", out_dir / "export.log")
    report: dict = {"category": args.category, "status": "unknown", "steps": []}

    def record(step: str, ok: bool, detail: str = "") -> None:
        report["steps"].append({"step": step, "ok": ok, "detail": detail})
        (log.info if ok else log.warning)("%s %s %s", "OK  " if ok else "FAIL", step, detail)

    ckpt = out_dir / "model.pt"
    if not ckpt.exists():
        log.error("缺少模型文件: %s，请先运行 train.py", ckpt)
        return 1

    run_cfg["runtime"]["resolved_device"] = "cpu"
    model, _ = load_checkpoint(run_cfg, args.category, ckpt)
    model.eval()
    # 导出原始 PatchCore 输出（关闭 MinMax 后处理），与 predict.py 使用的
    # 原始最近邻距离分数保持一致；阈值仍使用 results/<category>/threshold.json。
    model.post_processor = None
    # 导出契约：输入为已 Resize 到 256x256 的 [0,1] RGB 张量，图内只保留 Normalize。
    # 注意 Anomalib 的 export_transform 默认 Resize(antialias=False)，与训练用
    # transform(antialias=True) 不同；这里显式替换成仅 Normalize，保证 ONNX 与
    # predict.py 的预处理（先 Resize 再 Normalize）数值一致。
    from torchvision.transforms import v2

    from qc.modeling import IMAGENET_MEAN, IMAGENET_STD

    model.pre_processor.export_transform = v2.Compose([
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    record("加载模型", True, str(ckpt))

    export_root = out_dir / "export"
    export_root.mkdir(parents=True, exist_ok=True)
    onnx_path = export_root / "model.onnx"

    from anomalib.deploy import ExportType
    from qc.modeling import build_engine

    engine = build_engine(run_cfg, out_dir / "engine_export")
    exported = None
    last_error = None
    for attempt, extra in enumerate(({}, {"dynamo": False})):
        try:
            exported = engine.export(
                model=model,
                export_type=ExportType.ONNX,
                export_root=export_root,
                model_file_name="model",
                input_size=(int(run_cfg["data"]["image_size"][0]), int(run_cfg["data"]["image_size"][1])),
                onnx_kwargs={"opset_version": args.opset, **extra},
            )
            break
        except Exception as exc:  # 记录每次失败原因，换导出后端再试
            import traceback

            last_error = f"{type(exc).__name__}: {exc}"
            log.warning("ONNX 导出第 %d 次尝试失败: %s", attempt + 1, last_error)
            log.debug("%s", traceback.format_exc())
    if exported is None:
        report.update({"status": "export_failed", "error": last_error})
        record("ONNX 导出", False, last_error or "unknown")
        _write_report(out_dir, report)
        return 1
    exported = Path(exported)
    onnx_path = exported
    record("ONNX 导出", onnx_path.exists(), str(onnx_path))
    report["onnx_path"] = str(onnx_path)
    external = onnx_path.with_name(onnx_path.name + ".data")
    report["onnx_external_data"] = external.exists()
    total_bytes = onnx_path.stat().st_size + (external.stat().st_size if external.exists() else 0)
    report["onnx_size_mb"] = round(total_bytes / 1024**2, 2) if onnx_path.exists() else None

    # ---- ONNX Runtime 数值验证 ----
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_meta = sess.get_inputs()[0]
        out_names = [o.name for o in sess.get_outputs()]
        record("ONNX Runtime 加载", True, f"input={input_meta.name} {input_meta.shape} outputs={out_names}")

        data_root = Path(run_cfg["data"]["root"])
        if not data_root.is_absolute():
            data_root = (PROJECT_ROOT / data_root).resolve()
        sample = Path(args.sample) if args.sample else find_sample(out_dir, args.category, data_root)
        if sample is None or not sample.exists():
            record("数值验证", False, "找不到测试图片")
            report["status"] = "exported_unverified"
            _write_report(out_dir, report)
            return 0

        image = load_image_tensor(sample).unsqueeze(0)
        # 参考值走 predict.py 完全相同的路径：Resize(antialias) -> Normalize -> PatchCore，
        # 与 ONNX 图的输入契约（已 Resize 的 256 张量）一致。
        pt_score, pt_map = score_batch(model, image, run_cfg["data"]["image_size"], torch.device("cpu"))
        pt_score, pt_map = float(pt_score[0]), pt_map[0, 0].numpy()

        from torchvision.transforms import v2

        h, w = int(run_cfg["data"]["image_size"][0]), int(run_cfg["data"]["image_size"][1])
        resized = v2.Resize((h, w), antialias=True)(image)
        feeds = {
            "resized_256_raw_[0,1]": resized.numpy().astype(np.float32),
            "resized_256_imagenet_normalized": preprocess(image, run_cfg["data"]["image_size"]).numpy().astype(np.float32),
        }

        candidates = {}
        for label, feed in feeds.items():
            try:
                outs = sess.run(None, {input_meta.name: feed})
                out_map = {name: np.asarray(val) for name, val in zip(out_names, outs)}
                score_key = "pred_score" if "pred_score" in out_map else out_names[0]
                map_key = "anomaly_map" if "anomaly_map" in out_map else out_names[-1]
                score = float(out_map[score_key].reshape(-1)[0])
                amap = out_map[map_key].squeeze()
                candidates[label] = {
                    "score": score,
                    "score_abs_diff": abs(score - pt_score),
                    "map_shape": list(amap.shape),
                    "map_max_abs_diff": float(np.max(np.abs(amap - pt_map))) if amap.shape == pt_map.shape else None,
                }
            except Exception as exc:
                candidates[label] = {"error": f"{type(exc).__name__}: {exc}"}
        report["onnx_check"] = candidates
        report["pytorch_reference"] = {"score": pt_score, "map_shape": list(pt_map.shape),
                                       "score_space": "raw PatchCore distance (no post-processor)"}
        log.info("PyTorch 原始 PatchCore 参考分数: %.6f", pt_score)
        for label, c in candidates.items():
            if "error" in c:
                log.warning("  %s -> %s", label, c["error"])
            else:
                log.info("  %s -> score=%.6f (|diff|=%.6f), map_max_abs_diff=%s",
                         label, c["score"], c["score_abs_diff"], c["map_max_abs_diff"])

        best_label = None
        best = None
        for label, c in candidates.items():
            if "score_abs_diff" in c and (best is None or c["score_abs_diff"] < best["score_abs_diff"]):
                best, best_label = c, label
        if best and best["score_abs_diff"] < 1e-3 and (best.get("map_max_abs_diff") or 0) < 1e-2:
            report["status"] = "success"
            report["verified_input_format"] = best_label
            record("数值验证", True, f"最大分数误差 {best['score_abs_diff']:.2e}")
        else:
            report["status"] = "numerical_mismatch"
            record("数值验证", False, "ONNX 与 PyTorch 输出不一致")
    except Exception as exc:
        import traceback

        err = f"{type(exc).__name__}: {exc}"
        log.error("ONNX 验证失败: %s", err)
        log.debug("%s", traceback.format_exc())
        report.update({"status": "exported_unverified", "error": err})

    _write_report(out_dir, report)
    return 0


def _write_report(out_dir: Path, report: dict) -> None:
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# ONNX 导出报告 — {report['category']}",
        "",
        f"- 状态：**{report['status']}**",
    ]
    if report.get("onnx_path"):
        lines.append(f"- 导出文件：`{report['onnx_path']}`（合计 {report.get('onnx_size_mb')} MB"
                     f"{'，含外部权重文件 .data' if report.get('onnx_external_data') else ''}）")
    if report.get("error"):
        lines += ["", "## 报错", "", "```", report["error"], "```"]
    if report.get("onnx_check"):
        lines += ["", "## 数值验证（CPU，单张图）", "",
                  f"- PyTorch 参考分数（{report['pytorch_reference'].get('score_space', '')}）："
                  f"{report['pytorch_reference']['score']:.6f}",
                  f"- 验证通过的输入格式：`{report.get('verified_input_format', 'n/a')}`", "",
                  "| 输入格式 | ONNX 分数 | 与 PyTorch 差值 | 异常图最大绝对误差 |", "|---|---:|---:|---:|"]
        for k, v in report["onnx_check"].items():
            if "error" in v:
                lines.append(f"| {k} | 失败: {v['error'][:80]} | - | - |")
            else:
                lines.append(f"| {k} | {v['score']:.6f} | {v['score_abs_diff']:.3e} | {v['map_max_abs_diff']} |")
        lines += ["", "推理验证建议（ONNX Runtime，CPU）：", "",
                  "```python",
                  "import numpy as np, onnxruntime as ort",
                  "from PIL import Image",
                  "sess = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])",
                  "img = Image.open('part.png').convert('RGB').resize((256, 256), Image.BILINEAR)",
                  "x = (np.asarray(img, np.float32) / 255.0).transpose(2, 0, 1)[None]  # (1,3,256,256)",
                  "score, anomaly_map = sess.run(None, {sess.get_inputs()[0].name: x})",
                  "# score 为原始 PatchCore 距离，与 results/<category>/threshold.json 中的阈值比较得到 OK/NG",
                  "```"]
    lines += ["", "## 步骤日志", ""]
    for s in report.get("steps", []):
        lines.append(f"- [{'x' if s['ok'] else ' '}] {s['step']} {s['detail']}")
    (out_dir / "export_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
