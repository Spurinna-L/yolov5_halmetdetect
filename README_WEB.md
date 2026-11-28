# 智慧工地安全帽智能检测系统（WSL + ONNX Web 端）

## 1. 目录说明
- `backend/detector.py`：ONNX Runtime 推理引擎（Letterbox 前处理、NMS 后处理、OpenCV 绿/红框渲染、佩戴率与告警统计）
- `backend/main.py`：FastAPI 服务（图片检测接口、WebSocket 实时流式视频推理与静态页面托管）
- `frontend/`：VS Code 暗色工作台风格 Web 界面（图片双屏对比、流式视频实时推理与违规抓拍回看）
- `runs/weights/best.onnx`：由 Colab 训练的 YOLOv5s 导出得到的自包含 ONNX 模型

## 2. WSL 启动步骤
```bash
cd /mnt/d/vs_learning_code/yolov5_halmetdetect/project
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

如需单独指定权重路径，可通过环境变量设置：
```bash
export HELMET_MODEL_PATH=/mnt/d/vs_learning_code/yolov5_halmetdetect/project/runs/weights/best.onnx
```

## 3. 访问与使用
服务启动后，在 Windows 浏览器打开：
`http://localhost:8000`

1. **图片检测**：拖拽或点击上传图片，调节置信度 / IoU 滑块，查看原图与标注图对比、安全帽佩戴率及目标置信度明细。
2. **流式视频检测**：选择本地 MP4/MOV/WEBM 视频后零等待播放，浏览器按“背压闭环”抽帧并通过 WebSocket 发送到 WSL 推理，右侧实时画框与更新 FPS；连续 N 帧未佩戴安全帽触发告警并生成可点击跳转复查的违规抓拍卡片。
