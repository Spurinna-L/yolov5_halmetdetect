# -*- coding: utf-8 -*-
"""FastAPI backend serving the ONNX safety helmet detector, WebSocket stream, and web UI."""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.detector import HelmetDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_model_path() -> Path:
    """Find ONNX model path from environment variable or standard project locations."""
    env_path = os.environ.get("HELMET_MODEL_PATH")
    if env_path:
        return Path(env_path)
    candidates = [
        PROJECT_ROOT / "runs" / "weights" / "best.onnx",
        PROJECT_ROOT / "weights" / "best.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _prune_outputs(max_files: int = 80) -> None:
    """Keep the outputs directory bounded by removing oldest generated images."""
    files = sorted(
        (p for p in OUTPUTS_DIR.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    excess = len(files) - max_files
    for stale in files[: max(0, excess)]:
        try:
            stale.unlink()
        except OSError:
            pass


class AppState:
    detector: HelmetDetector | None = None
    detector_error: str | None = None
    model_path: Path = _resolve_model_path()


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load ONNX model once at startup."""
    state.model_path = _resolve_model_path()
    try:
        state.detector = HelmetDetector(state.model_path)
        state.detector_error = None
    except Exception as exc:  # noqa: BLE001
        state.detector = None
        state.detector_error = str(exc)
    yield


app = FastAPI(
    title="安全帽检测工作台",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the single-page web monitoring console."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    """Return backend health and loaded ONNX model metadata."""
    if state.detector is None:
        return {
            "ready": False,
            "model_name": state.model_path.name,
            "model_path": str(state.model_path),
            "input_size": 640,
            "providers": [],
            "classes": [],
            "error": state.detector_error or "Model not loaded",
        }

    return {
        "ready": True,
        "model_name": state.detector.model_path.name,
        "model_path": str(state.detector.model_path),
        "input_size": state.detector.img_size,
        "providers": state.detector.providers,
        "classes": state.detector.class_names,
        "error": None,
    }


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.40),
    iou: float = Form(0.45),
    draw_hud: bool = Form(True),
) -> dict[str, Any]:
    """Run ONNX detection on an uploaded image and return annotated result URL + KPIs."""
    if state.detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"推理模型未就绪: {state.detector_error or '请检查 ONNX 权重路径'}",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="上传的文件为空")

    img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="无法解析该图片文件，请上传 JPG/PNG/BMP/WEBP 格式")

    conf_clamped = float(np.clip(conf, 0.05, 0.95))
    iou_clamped = float(np.clip(iou, 0.05, 0.95))

    annotated, summary = state.detector.detect_and_render(
        img_bgr,
        conf_thres=conf_clamped,
        iou_thres=iou_clamped,
        draw_hud=draw_hud,
    )

    out_name = f"det_{uuid4().hex[:10]}.jpg"
    out_path = OUTPUTS_DIR / out_name
    cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    _prune_outputs(max_files=80)

    height, width = img_bgr.shape[:2]
    return {
        "output_url": f"/outputs/{out_name}",
        "filename": out_name,
        "width": int(width),
        "height": int(height),
        **summary,
    }


@app.websocket("/ws/stream")
async def stream_detect(websocket: WebSocket) -> None:
    """Real-time frame-by-frame WebSocket endpoint for streaming video inspection."""
    await websocket.accept()
    if state.detector is None:
        await websocket.send_json(
            {
                "type": "error",
                "message": f"推理模型未就绪: {state.detector_error or '未找到 best.onnx'}",
            }
        )
        await websocket.close()
        return

    try:
        while True:
            payload = await websocket.receive_json()
            raw_image = payload.get("image", "")
            if not raw_image:
                await websocket.send_json({"type": "error", "message": "缺少图像帧数据"})
                continue

            if "," in raw_image:
                raw_image = raw_image.split(",", 1)[1]

            try:
                frame_bytes = base64.b64decode(raw_image)
                frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001
                frame_bgr = None

            if frame_bgr is None:
                await websocket.send_json({"type": "error", "message": "视频帧解码失败"})
                continue

            conf = float(np.clip(float(payload.get("conf", 0.40)), 0.05, 0.95))
            iou = float(np.clip(float(payload.get("iou", 0.45)), 0.05, 0.95))
            draw_hud = bool(payload.get("draw_hud", True))
            timestamp = payload.get("timestamp", 0.0)

            annotated, summary = state.detector.detect_and_render(
                frame_bgr,
                conf_thres=conf,
                iou_thres=iou,
                draw_hud=draw_hud,
            )

            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                await websocket.send_json({"type": "error", "message": "结果帧编码失败"})
                continue

            b64_out = base64.b64encode(encoded.tobytes()).decode("ascii")
            await websocket.send_json(
                {
                    "type": "result",
                    "annotated_image": f"data:image/jpeg;base64,{b64_out}",
                    "timestamp": timestamp,
                    **summary,
                }
            )
    except WebSocketDisconnect:
        pass


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
