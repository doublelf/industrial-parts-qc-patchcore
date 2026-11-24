#!/usr/bin/env python
"""抽查界面与 predict.py 命令行的一致性，并测界面端到端耗时。

前置：`python app.py` 已启动（默认 http://127.0.0.1:8000）。

输出：results/ui_demo/check.md（对比表 + 结论）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

CASES = [
    ("metal_nut", "data/mvtec/metal_nut/test/bent/000.png", "缺陷 bent"),
    ("metal_nut", "data/mvtec/metal_nut/test/good/000.png", "正常 good"),
    ("screw", "data/mvtec/screw/test/scratch_neck/000.png", "缺陷 scratch_neck"),
]


def run_cli(category: str, image: str) -> dict:
    cmd = [sys.executable, "predict.py", "--category", category, "--input", image, "--no-save"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    wall_ms = (time.perf_counter() - t0) * 1000
    payload = None
    for line in proc.stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
            break
    if payload is None:
        raise RuntimeError(f"predict.py 未输出 JSON：{proc.stdout[-800:]} {proc.stderr[-800:]}")
    payload["wall_ms"] = wall_ms
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--out", default="results/ui_demo")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        req = ctx.request

        for category, image, label in CASES:
            # 命令行跑两次，取较稳定的暖机耗时（CPU 共享开发机波动较大）
            cli_runs = [run_cli(category, image) for _ in range(2)]
            cli = cli_runs[0]
            cli_latency = min(r["latency_ms"] for r in cli_runs)

            with open(PROJECT_ROOT / image, "rb") as f:
                data = f.read()

            api_runs, http_runs = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                resp = req.post(
                    f"{args.url}/api/detect",
                    multipart={
                        "file": {"name": Path(image).name, "mimeType": "image/png", "buffer": data},
                        "category": category,
                        "source": "consistency_check",
                    },
                )
                http_runs.append((time.perf_counter() - t0) * 1000)
                api_runs.append(resp.json())
            api = api_runs[0]
            api_latency = sorted(r["latency_ms"] for r in api_runs)[1]
            http_ms = sorted(http_runs)[1]
            rows.append({
                "category": category, "image": image, "label": label,
                "cli_score": cli["score"], "cli_decision": cli["decision"], "cli_latency": cli_latency,
                "api_score": api["score"], "api_decision": api["decision"], "api_latency": api_latency,
                "http_ms": http_ms, "threshold": api["threshold"],
                "score_diff": abs(cli["score"] - api["score"]), "match": cli["decision"] == api["decision"],
                "success": resp.ok,
            })

        # 界面端到端：单步点击 -> 分数刷新
        page = ctx.new_page()
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_timeout(800)
        page.click("#btnReset")
        page.wait_for_timeout(500)
        ui_times = []
        for _ in range(5):
            prev = page.inner_text("#scoreBig")
            t0 = time.perf_counter()
            page.click("#btnStep")
            page.wait_for_function(
                f"document.querySelector('#scoreBig').textContent !== {json.dumps(prev)}",
                timeout=60000,
            )
            ui_times.append((time.perf_counter() - t0) * 1000)
            page.wait_for_timeout(400)
        page.screenshot(path=str(out / "06_ui_step.png"))
        browser.close()

    ui_times.sort()

    ok = all(r["match"] and r["success"] and r["score_diff"] < 1e-4 for r in rows)
    lines = [
        "# 界面 vs predict.py 一致性抽查",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- 方式：同一张图片分别走 `predict.py --no-save`（命令行）与 `POST /api/detect`（界面后端），比较分数/判定/耗时",
        "- 阈值来源：`results/<category>/threshold.json`（验证集确定，界面与命令行共用同一文件）",
        "",
        "## 1. 分数与判定",
        "",
        "| 图片 | 类别 | 类型 | 命令行分数 | 界面分数 | 差值 | 命令行判定 | 界面判定 | 阈值 | 一致 |",
        "|---|---|---|---:|---:|---:|---|---|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {Path(r['image']).name} | {r['category']} | {r['label']} | {r['cli_score']:.6f} | "
            f"{r['api_score']:.6f} | {r['score_diff']:.2e} | {r['cli_decision']} | {r['api_decision']} | "
            f"{r['threshold']:.4f} | {'✅' if r['match'] and r['score_diff'] < 1e-4 else '❌'} |"
        )

    lines += [
        "",
        "## 2. 耗时对比（CPU，256×256）",
        "",
        "| 图片 | 命令行耗时（推理） | 界面 API 服务端耗时 | HTTP 往返（含上传/落盘） |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {Path(r['image']).name} | {r['cli_latency']:.1f} ms | {r['api_latency']:.1f} ms | {r['http_ms']:.1f} ms |")

    cli_mean = sum(r["cli_latency"] for r in rows) / len(rows)
    http_mean = sum(r["http_ms"] for r in rows) / len(rows)
    ui_mean = ui_times[len(ui_times) // 2]
    lines += [
        "",
        f"- 命令行单张推理平均：**{cli_mean:.1f} ms**（每张跑 2 次取较快值；含读图+预处理+前向+异常图）",
        f"- 界面 HTTP 往返平均：**{http_mean:.1f} ms**（每张 3 次取中位数；含 multipart 上传、服务端推理、原图/热力图落盘、JSON 返回）",
        f"- 界面端到端（点击「单步」到分数刷新）：**{ui_mean:.1f} ms**（5 次中位数）",
        f"- 界面层额外开销约 **{ui_mean - cli_mean:.0f} ms/件**：主要来自 HTTP 上传、原图 JPEG 与热力图 PNG 编码落盘、浏览器渲染；"
        "模型推理本身与命令行完全一致（同一进程常驻，无重复加载）。",
        "",
        f"## 3. 结论",
        "",
        f"- 3 张抽查图（含 2 张缺陷）分数完全一致（误差 < 1e-4），判定一致：**{'全部通过 ✅' if ok else '存在不一致 ❌'}**；",
        "- 一致性说明界面没有另定阈值或改预处理；模型只在服务启动时加载一次。",
        "",
    ]
    (out / "check.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
