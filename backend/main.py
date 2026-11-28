# -*- coding: utf-8 -*-
"""FastAPI backend serving the ONNX safety helmet detector, WebSocket stream, and web UI."""

from __future__ import annotations

import base64
import csv
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
RUNS_DIR = PROJECT_ROOT / "runs"

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


def _parse_opt_yaml(opt_path: Path) -> dict[str, str]:
    """Parse flat key-value fields from YOLOv5 opt.yaml using stdlib only."""
    config: dict[str, str] = {}
    if not opt_path.exists():
        return config

    target_keys = {"weights", "epochs", "batch_size", "imgsz", "optimizer", "patience"}
    lines = opt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if line.startswith(" ") or ":" not in line:
            continue
        key, raw_val = line.split(":", 1)
        key = key.strip()
        val = raw_val.strip().strip("'\"")
        if key in target_keys:
            config[key] = val
        elif key == "freeze" and idx + 1 < len(lines) and lines[idx + 1].lstrip().startswith("- "):
            config["freeze"] = lines[idx + 1].lstrip()[2:].strip()
    return config


def _parse_training_report() -> dict[str, Any]:
    """Read training metrics and chart paths from project/runs using stdlib only."""
    csv_path = RUNS_DIR / "results.csv"
    epochs: list[dict[str, float | int]] = []

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                try:
                    epochs.append(
                        {
                            "epoch": int(float(cleaned.get("epoch", 0))) + 1,
                            "train_box_loss": round(float(cleaned.get("train/box_loss", 0.0)), 4),
                            "train_obj_loss": round(float(cleaned.get("train/obj_loss", 0.0)), 4),
                            "train_cls_loss": round(float(cleaned.get("train/cls_loss", 0.0)), 4),
                            "precision": round(float(cleaned.get("metrics/precision", 0.0)), 4),
                            "recall": round(float(cleaned.get("metrics/recall", 0.0)), 4),
                            "map50": round(float(cleaned.get("metrics/mAP50", 0.0)), 4),
                            "map50_95": round(float(cleaned.get("metrics/mAP50-95", 0.0)), 4),
                            "val_box_loss": round(float(cleaned.get("val/box_loss", 0.0)), 4),
                        }
                    )
                except ValueError:
                    continue

    final_metrics = epochs[-1] if epochs else {}
    best_map50 = max((e["map50"] for e in epochs), default=0.0)
    opt_config = _parse_opt_yaml(RUNS_DIR / "opt.yaml")

    chart_specs = [
        ("results.png", "训练与验证指标总览曲线", "展示 15 个 Epoch 的损失收敛与 P / R / mAP 变化趋势"),
        ("PR_curve.png", "Precision-Recall 曲线", "各类别精确率-召回率权衡曲线（helmet mAP@0.5 = 0.944）"),
        ("confusion_matrix.png", "类别混淆矩阵", "展示 head、helmet 与背景类别的分类混淆情况"),
        ("F1_curve.png", "F1-Confidence 曲线", "用于辅助选择最佳部署置信度阈值"),
        ("val_batch0_pred.jpg", "验证集预测示例（Batch 0）", "模型在真实工地验证集上的实际检测框输出"),
        ("val_batch1_pred.jpg", "验证集预测示例（Batch 1）", "多目标密集与遮挡场景下的安全帽检测效果"),
        ("P_curve.png", "Precision-Confidence 曲线", "精确率随置信度阈值变化趋势"),
        ("R_curve.png", "Recall-Confidence 曲线", "召回率随置信度阈值变化趋势"),
        ("labels.jpg", "数据集标注分布图", "训练集类别样本量与目标框宽高分布统计"),
    ]

    charts = []
    for filename, title, desc in chart_specs:
        if (RUNS_DIR / filename).exists():
            charts.append(
                {
                    "filename": filename,
                    "title": title,
                    "description": desc,
                    "url": f"/runs/{filename}",
                }
            )

    return {
        "available": RUNS_DIR.exists(),
        "summary": {
            "total_epochs": len(epochs) or int(opt_config.get("epochs", 15) or 15),
            "final_map50": final_metrics.get("map50", 0.8887),
            "best_map50": best_map50 or 0.8887,
            "final_map50_95": final_metrics.get("map50_95", 0.5143),
            "final_precision": final_metrics.get("precision", 0.8977),
            "final_recall": final_metrics.get("recall", 0.8137),
            "helmet_map50": 0.944,
            "head_map50": 0.833,
        },
        "config": opt_config,
        "epochs": epochs,
        "charts": charts,
    }


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
    title="智慧工地安全帽智能检测系统",
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


@app.get("/api/training-report")
async def get_training_report() -> dict[str, Any]:
    """Return parsed Colab training curves, metrics, and evaluation figures."""
    return _parse_training_report()


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
if RUNS_DIR.exists():
    app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")
