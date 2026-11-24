#!/usr/bin/env python
"""用 Playwright 真实驱动演示界面，跑产线模拟并截图。

前置：先启动服务 `python app.py`（默认 http://127.0.0.1:8000）。

用法：
    python scripts/ui_demo_run.py --items 24 --interval 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="演示界面自动化验证与截图")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--items", type=int, default=24, help="产线模式至少检测的件数")
    p.add_argument("--interval", type=float, default=0.5, help="界面节拍（秒）")
    p.add_argument("--out", default="results/ui_demo")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 清理上一轮截图，避免旧文件与本次报告混淆
    for old in out.glob("0*.png"):
        old.unlink()
    console_errors: list[str] = []
    report: dict = {"items_target": args.items, "interval_s": args.interval, "screenshots": []}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome", headless=True,
            args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.on("console", lambda m: console_errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

        page.goto(args.url, wait_until="networkidle", timeout=60000)
        page.wait_for_function("document.querySelectorAll('#category option').length > 0", timeout=30000)

        # 干净起点
        page.click("#btnReset")
        page.wait_for_timeout(600)
        page.screenshot(path=str(out / "01_line_idle.png"))
        report["screenshots"].append("01_line_idle.png")

        # 设置节拍
        page.eval_on_selector("#interval", f"el => {{ el.value = '{args.interval}'; el.dispatchEvent(new Event('input')); }}")
        page.wait_for_timeout(200)

        t0 = time.perf_counter()
        page.click("#btnStart")
        # 第一件 NG 出现时抓一张（判定灯变红）
        ng_shot = False
        try:
            page.wait_for_selector("#lamp.ng", timeout=180000)
            page.wait_for_timeout(250)
            page.screenshot(path=str(out / "02_line_ng_moment.png"))
            report["screenshots"].append("02_line_ng_moment.png")
            ng_shot = True
        except Exception:
            report["ng_moment_captured"] = False

        # 等累计检测数达到目标
        page.wait_for_function(
            f"parseInt(document.querySelector('#statTotal').textContent, 10) >= {args.items}",
            timeout=240000,
        )
        elapsed = time.perf_counter() - t0
        page.click("#btnPause")
        page.wait_for_timeout(500)
        page.screenshot(path=str(out / "03_line_items.png"))
        report["screenshots"].append("03_line_items.png")

        stats = {
            "total": page.inner_text("#statTotal"),
            "ok": page.inner_text("#statOk"),
            "ng": page.inner_text("#statNg"),
            "defect_rate": page.inner_text("#statRate"),
            "avg_latency": page.inner_text("#statLatency"),
            "table_rows": page.eval_on_selector_all("#histBody tr", "els => els.length"),
            "trend_points": page.evaluate("document.querySelector('#trendCanvas').width"),
        }
        report.update({"line_stats": stats, "elapsed_s": round(elapsed, 1), "ng_shot": ng_shot})

        # 摄像头模式（Chrome fake device）
        try:
            page.click("#modeCamera")
            camera_ok = False
            for _ in range(20):  # 最多等 10 秒让 fake camera 出画
                page.wait_for_timeout(500)
                if page.eval_on_selector("#video", "el => !el.hidden && el.videoWidth > 0"):
                    camera_ok = True
                    break
            if camera_ok:
                # 等首帧不再全黑（fake device 启动初期可能是黑帧）
                page.wait_for_function(
                    """() => {
                        const v = document.querySelector('#video');
                        if (!v.videoWidth) return false;
                        const c = document.createElement('canvas');
                        c.width = 64; c.height = 36;
                        const ctx = c.getContext('2d');
                        ctx.drawImage(v, 0, 0, 64, 36);
                        const d = ctx.getImageData(0, 0, 64, 36).data;
                        let s = 0;
                        for (let i = 0; i < d.length; i += 4) s += d[i] + d[i+1] + d[i+2];
                        return s / (d.length / 4) / 3 > 15;
                    }""",
                    timeout=15000,
                )
                page.screenshot(path=str(out / "04_camera_mode.png"))
                report["screenshots"].append("04_camera_mode.png")
                page.click("#btnShoot")
                page.wait_for_timeout(2500)
                page.screenshot(path=str(out / "05_camera_capture.png"))
                report["screenshots"].append("05_camera_capture.png")
                report["camera_mode"] = "ok"
                report["camera_stats"] = {
                    "total": page.inner_text("#statTotal"),
                    "ng": page.inner_text("#statNg"),
                    "last_score": page.inner_text("#scoreBig"),
                }
                # 摄像头验证结束后切回产线模式再检测一件，保证刷新截图展示的是真实零件
                page.click("#modeLine")
                page.wait_for_timeout(800)
                prev = page.inner_text("#scoreBig")
                page.click("#btnStep")
                page.wait_for_function(
                    f"document.querySelector('#scoreBig').textContent !== {json.dumps(prev)}",
                    timeout=60000,
                )
                page.wait_for_timeout(1200)
            else:
                page.screenshot(path=str(out / "04_camera_fallback.png"))
                report["screenshots"].append("04_camera_fallback.png")
                report["camera_mode"] = "unavailable_fallback"
        except Exception as exc:
            report["camera_mode"] = f"error: {type(exc).__name__}: {exc}"
            try:
                page.screenshot(path=str(out / "04_camera_fallback.png"))
            except Exception:
                pass

        # 刷新页面：统计应从服务端恢复（接着累计），而不是清零
        total_before = page.inner_text("#statTotal")
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(1200)
        report["after_reload"] = {
            "total_before": total_before,
            "total_after": page.inner_text("#statTotal"),
            "ng_after": page.inner_text("#statNg"),
            "history_rows_after": page.eval_on_selector_all("#histBody tr", "els => els.length"),
            "counter_mode": "服务端累计，刷新后接着累计",
        }
        page.screenshot(path=str(out / "07_after_reload.png"))
        report["screenshots"].append("07_after_reload.png")

        report["console_errors"] = console_errors[:10]
        browser.close()

    csv_path = Path("outputs/live/live_log.csv")
    if csv_path.exists():
        lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
        report["live_log_rows"] = len(lines) - 1
        report["live_log_path"] = str(csv_path)
        report["live_log_header"] = lines[0] if lines else ""
    (out / "ui_run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
