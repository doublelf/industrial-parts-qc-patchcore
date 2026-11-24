#!/usr/bin/env python
"""工业零件质检演示界面（FastAPI + 原生 HTML/JS）。

启动：
    python app.py                 # 默认 http://127.0.0.1:8000
    python app.py --port 8080

说明：
- 模型在服务启动时一次性加载并常驻内存，请求内不再重新加载；
- 阈值来自 results/<category>/threshold.json（验证集确定），界面只展示不修改；
- 这是模拟演示界面，图片来自演示文件夹/MVTec test 集，不接真实产线。
"""
from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from qc.config import load_config, resolve_path  # noqa: E402

WEB_DIR = PROJECT_ROOT / "web"
CFG = load_config()
LIVE_DIR = resolve_path(CFG["ui"]["live_dir"])
LIVE_DIR.mkdir(parents=True, exist_ok=True)
service = None  # 由 lifespan 初始化


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时加载全部已训练类别（每个模型只加载一次，常驻内存）。"""
    global service
    from qc.live_service import LiveQCService

    service = LiveQCService()
    yield


app = FastAPI(title="Industrial Parts QC Demo", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
app.mount("/live", StaticFiles(directory=str(LIVE_DIR)), name="live")


class DemoRequest(BaseModel):
    category: str | None = None


def _svc():
    if service is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    return service


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    svc = _svc()
    return {"status": "ok", "categories": list(svc.entries.keys()), "device": str(svc.device)}


@app.get("/api/state")
def state(category: str | None = None) -> dict:
    try:
        return _svc().state(category)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/stats")
def stats() -> dict:
    return _svc().stats()


@app.post("/api/demo/next")
def demo_next(req: DemoRequest) -> dict:
    try:
        return _svc().next_demo(req.category)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), category: str = Form(...),
                 source: str = Form("upload")) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        return _svc().detect_bytes(data, category, file.filename or "upload.jpg",
                                   mode="camera" if source == "camera" else "upload", source=source)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # 图片解码失败等
        raise HTTPException(status_code=400, detail=f"无法处理该图片: {type(exc).__name__}: {exc}") from exc


@app.post("/api/reset")
def reset() -> dict:
    return _svc().reset()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="质检演示界面")
    parser.add_argument("--host", default=CFG["ui"].get("host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(CFG["ui"].get("port", 8000)))
    args = parser.parse_args()
    print("=" * 64)
    print(f"  工业零件质检演示界面: http://{args.host}:{args.port}")
    print("  提示：这是模拟演示，不接真实产线；Ctrl+C 停止")
    print("=" * 64)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
