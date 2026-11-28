# -*- coding: utf-8 -*-
"""YOLOv5 ONNX safety helmet detector with OpenCV rendering and safety statistics."""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_CLASS_NAMES = ["person", "head", "helmet"]

CN_LABEL_MAP = {
    "helmet": "已佩戴安全帽",
    "head": "未佩戴安全帽",
    "person": "人员",
}

ASCII_LABEL_MAP = {
    "helmet": "HELMET",
    "head": "NO-HELMET",
    "person": "PERSON",
}

# BGR colors for OpenCV drawing
COLOR_MAP = {
    "helmet": (80, 210, 60),   # Green (compliant)
    "head": (60, 60, 245),     # Red (violation)
    "person": (0, 185, 255),   # Amber
}


class HelmetDetector:
    """Self-contained ONNX Runtime detector for YOLOv5 safety helmet models."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        available_providers = ort.get_available_providers()
        preferred_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=preferred_providers,
        )
        self.providers = self.session.get_providers()

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = model_input.shape
        raw_h = model_input.shape[2] if len(model_input.shape) >= 4 else 640
        self.img_size = int(raw_h) if isinstance(raw_h, int) and raw_h > 0 else 640

        self.output_name = self.session.get_outputs()[0].name
        self.class_names = class_names or self._extract_class_names()

        # Warmup inference once so the first user request has minimal latency.
        dummy = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

    def _extract_class_names(self) -> list[str]:
        """Extract class names from ONNX metadata if present, otherwise use default."""
        meta = self.session.get_modelmeta()
        raw_names = meta.custom_metadata_map.get("names")
        if not raw_names:
            return list(DEFAULT_CLASS_NAMES)
        try:
            parsed = ast.literal_eval(raw_names)
            if isinstance(parsed, dict):
                return [str(parsed[k]) for k in sorted(parsed.keys())]
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError, TypeError):
            pass
        return list(DEFAULT_CLASS_NAMES)

    def letterbox(
        self,
        img_bgr: np.ndarray,
        new_shape: int | None = None,
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize image keeping aspect ratio and pad to square new_shape x new_shape."""
        target = new_shape or self.img_size
        orig_h, orig_w = img_bgr.shape[:2]

        ratio = min(target / orig_h, target / orig_w)
        new_unpad_w = int(round(orig_w * ratio))
        new_unpad_h = int(round(orig_h * ratio))

        dw = (target - new_unpad_w) / 2.0
        dh = (target - new_unpad_h) / 2.0

        if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
            interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(img_bgr, (new_unpad_w, new_unpad_h), interpolation=interp)
        else:
            resized = img_bgr

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        # Ensure exact target dimensions even on odd rounding edge cases.
        pad_h_diff = target - (new_unpad_h + top + bottom)
        pad_w_diff = target - (new_unpad_w + left + right)
        bottom += pad_h_diff
        right += pad_w_diff

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )
        return padded, ratio, (left, top)

    def preprocess(
        self,
        img_bgr: np.ndarray,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Convert BGR image into NCHW float32 tensor normalized to [0, 1]."""
        padded, ratio, pad = self.letterbox(img_bgr, new_shape=self.img_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob), ratio, pad

    def postprocess(
        self,
        pred: np.ndarray,
        orig_shape: tuple[int, int],
        ratio: float,
        pad: tuple[int, int],
        conf_thres: float,
        iou_thres: float,
    ) -> list[dict[str, Any]]:
        """Filter raw YOLOv5 predictions, run class-agnostic NMS, and rescale boxes."""
        if pred.ndim == 3:
            pred = pred[0]
        if pred.size == 0:
            return []

        obj_mask = pred[:, 4] > conf_thres
        candidates = pred[obj_mask]
        if candidates.shape[0] == 0:
            return []

        cls_scores = candidates[:, 5:] * candidates[:, 4:5]
        class_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(candidates.shape[0]), class_ids]

        keep_mask = scores >= conf_thres
        candidates = candidates[keep_mask]
        class_ids = class_ids[keep_mask]
        scores = scores[keep_mask]

        if candidates.shape[0] == 0:
            return []

        orig_h, orig_w = orig_shape
        pad_w, pad_h = pad

        cx = candidates[:, 0]
        cy = candidates[:, 1]
        bw = candidates[:, 2]
        bh = candidates[:, 3]

        x1 = np.clip((cx - bw / 2.0 - pad_w) / ratio, 0, orig_w)
        y1 = np.clip((cy - bh / 2.0 - pad_h) / ratio, 0, orig_h)
        x2 = np.clip((cx + bw / 2.0 - pad_w) / ratio, 0, orig_w)
        y2 = np.clip((cy + bh / 2.0 - pad_h) / ratio, 0, orig_h)

        # cv2.dnn.NMSBoxes expects [x, y, w, h] list of floats.
        xywh_boxes = np.stack([x1, y1, np.maximum(0.0, x2 - x1), np.maximum(0.0, y2 - y1)], axis=1).tolist()
        score_list = scores.astype(float).tolist()

        # Class-agnostic NMS avoids reporting the same head as both `head` and `helmet`.
        indices = cv2.dnn.NMSBoxes(xywh_boxes, score_list, float(conf_thres), float(iou_thres))
        if indices is None or len(indices) == 0:
            return []

        kept_indices = np.asarray(indices).reshape(-1).tolist()

        detections: list[dict[str, Any]] = []
        for idx in kept_indices:
            cls_id = int(class_ids[idx])
            cls_name = (
                self.class_names[cls_id]
                if 0 <= cls_id < len(self.class_names)
                else f"class_{cls_id}"
            )
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "label_cn": CN_LABEL_MAP.get(cls_name, cls_name),
                    "confidence": round(float(scores[idx]), 4),
                    "bbox": [
                        int(round(x1[idx])),
                        int(round(y1[idx])),
                        int(round(x2[idx])),
                        int(round(y2[idx])),
                    ],
                }
            )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def summarize(self, detections: list[dict[str, Any]], infer_ms: float) -> dict[str, Any]:
        """Compute safety compliance KPIs from filtered detections."""
        helmet_count = sum(1 for d in detections if d["class_name"] == "helmet")
        head_count = sum(1 for d in detections if d["class_name"] == "head")
        total = helmet_count + head_count
        compliance_rate = round((helmet_count / total) * 100.0, 1) if total > 0 else None
        is_alarm = head_count > 0
        rounded_ms = round(float(infer_ms), 2)
        fps = round(1000.0 / max(infer_ms, 0.1), 1)

        return {
            "total": total,
            "helmet_count": helmet_count,
            "head_count": head_count,
            "compliance_rate": compliance_rate,
            "is_alarm": is_alarm,
            "infer_ms": rounded_ms,
            "fps": fps,
            "detections": detections,
        }

    def render(
        self,
        img_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        summary: dict[str, Any],
        draw_hud: bool = True,
    ) -> np.ndarray:
        """Draw crisp bounding boxes, ASCII tags, and optional HUD onto a copy of the image."""
        canvas = img_bgr.copy()
        img_h, img_w = canvas.shape[:2]
        thickness = max(2, int(round(0.0028 * (img_h + img_w) / 2.0)))
        font_scale = max(0.5, min(0.95, thickness * 0.28))

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det["class_name"]
            color = COLOR_MAP.get(cls_name, (0, 185, 255))
            tag = f"{ASCII_LABEL_MAP.get(cls_name, cls_name.upper())} {det['confidence']:.2f}"

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

            (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1))
            label_h = th + baseline + 8
            top_y = max(0, y1 - label_h)
            right_x = min(img_w, x1 + tw + 10)
            cv2.rectangle(canvas, (x1, top_y), (right_x, top_y + label_h), color, -1)

            text_y = top_y + th + 4
            cv2.putText(
                canvas,
                tag,
                (x1 + 5, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                max(1, thickness - 1),
                lineType=cv2.LINE_AA,
            )

        if draw_hud:
            status_text = "ALARM" if summary["is_alarm"] else "SAFE"
            hud_text = (
                f"Helmet: {summary['helmet_count']} | "
                f"No-Helmet: {summary['head_count']} | "
                f"{status_text}"
            )
            hud_scale = max(0.55, min(0.85, img_w / 1100.0))
            hud_thick = 2 if hud_scale >= 0.65 else 1
            (htw, hth), hbase = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, hud_scale, hud_thick)
            box_w = min(img_w - 16, htw + 24)
            box_h = hth + hbase + 18

            overlay = canvas.copy()
            cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (18, 22, 30), -1)
            cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

            border_color = (60, 60, 245) if summary["is_alarm"] else (80, 210, 60)
            cv2.rectangle(canvas, (10, 10), (10 + box_w, 10 + box_h), border_color, 2)
            cv2.putText(
                canvas,
                hud_text,
                (22, 10 + hth + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                hud_scale,
                (245, 248, 255),
                hud_thick,
                lineType=cv2.LINE_AA,
            )

        return canvas

    def detect_and_render(
        self,
        img_bgr: np.ndarray,
        conf_thres: float = 0.40,
        iou_thres: float = 0.45,
        draw_hud: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run end-to-end preprocessing, ONNX inference, NMS, summarization, and rendering."""
        t_start = time.perf_counter()
        blob, ratio, pad = self.preprocess(img_bgr)
        raw_out = self.session.run([self.output_name], {self.input_name: blob})[0]
        detections = self.postprocess(
            raw_out,
            img_bgr.shape[:2],
            ratio,
            pad,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        infer_ms = (time.perf_counter() - t_start) * 1000.0
        summary = self.summarize(detections, infer_ms)
        annotated = self.render(img_bgr, detections, summary, draw_hud=draw_hud)
        return annotated, summary
