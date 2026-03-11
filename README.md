# 🔥 Real-Time Fire & Smoke Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/YOLOv5-PyTorch-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/ONNX-Runtime-lightgrey?style=for-the-badge&logo=onnx" />
  <img src="https://img.shields.io/badge/WebSocket-Realtime-orange?style=for-the-badge" />
</p>

<p align="center">
  A blazing-fast, browser-based real-time fire and smoke detection system powered by a custom-trained YOLOv5 model, served via FastAPI with WebSocket streaming.
</p>

---

## 📸 Demo

| 🔥 Fire Detection | 💨 Smoke Detection |
|---|---|
| Bounding box drawn in **red** with confidence % | Bounding box drawn in **white** with confidence % |

> The app runs live through your webcam or accepts uploaded images — no GPU required!

---

## ✨ Features

- ⚡ **Real-time detection** via webcam using WebSocket for ultra-low latency streaming
- 🖼️ **Static image upload** support for single-frame analysis
- 🔍 Detects **3 classes**: `fire`, `smoke`, `other`
- 📊 **Live confidence scores** and detection counts per class
- 📷 **Front / Back camera toggle** for mobile devices
- 🧠 **Smart detection feed** — logs an alert only after 10 consecutive frames confirm detection (reduces false positives)
- 📥 **Downloadable annotated images** from both live feed snapshots and uploaded images
- 🚀 Powered by **ONNX Runtime** — fast CPU inference, no GPU needed
- 🌐 Clean, responsive **Bootstrap 5** UI

---

## 🔧 Prerequisites

Make sure you have the following installed:

- **Python 3.9 or higher** → [Download Python](https://www.python.org/downloads/)
- **pip** (comes with Python)
- A working **webcam** (optional — for live detection mode)

> ℹ️ No GPU is required. The model runs on CPU via ONNX Runtime.

---

## 🚀 Installation

### 1. Clone or Download the Project

```bash
git clone https://github.com/your-username/fire-smoke-detection.git
cd fire-smoke-detection
```

Or simply download and extract the ZIP, then navigate to the folder.

---

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv env
env\Scripts\activate

# macOS / Linux
python3 -m venv env
source env/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note for Windows users:** PyTorch CPU-only wheels are fetched from the PyTorch index. If you encounter issues, run:
> ```bash
> pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
> ```

---

### 4. Verify Model Files Are Present

Ensure both model files exist in the project root:
- `fire_n_smoke.onnx` ← used for inference
- `fire_n_smoke.pt` ← original PyTorch weights (backup)

---

## ▶️ Running the App

Start the FastAPI server with:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser and navigate to:

```
http://localhost:8000
```

---

## 🖥️ Usage Guide

### 📷 Live Webcam Detection

1. Open `http://localhost:8000` in your browser
2. Click **"📷 Enable Camera"**
3. Grant camera permissions when prompted
4. Detection begins automatically — bounding boxes appear in real time on the canvas

| Button | Action |
|---|---|
| `📷 Enable Camera` | Starts live webcam feed + WebSocket detection |
| `🚫 Disable Camera` | Stops camera and WebSocket connection |
| `🔄 Switch Camera` | Toggles between front and back camera (mobile) |

> 🧠 The **Detection Feed** panel logs confirmed detections — a label is only logged after appearing in **10 consecutive frames**, reducing noise from false positives.

---

### 🖼️ Static Image Upload

1. Click **"Choose File"** and select any image (JPG, PNG, etc.)
2. The image is resized to 640×640 and sent to the backend
3. Detection results appear as annotated bounding boxes on the canvas
4. Click **"Download Annotated Image"** to save the result

---

### 📊 Detection Dashboard

The top of the page shows live counters and average confidence per class:

| Badge | Description |
|---|---|
| 🔥 Fire | Count + average confidence across fire detections |
| 💨 Smoke | Count + average confidence across smoke detections |
| ✅ Other | Count + average confidence for non-threat objects |

---

### 📥 Downloading Snapshots

- **From webcam:** Each confirmed detection in the feed includes a snapshot thumbnail and a `📥 Download Annotated Image` link
- **From image upload:** A download button appears after analysis completes

---

## 🧠 How It Works

```
Browser (Webcam)
     │
     │  Base64 JPEG frame (every 250ms)
     ▼
FastAPI WebSocket (/ws)
     │
     │  Preprocess: Resize → Normalize → CHW format
     ▼
ONNX Runtime (fire_n_smoke.onnx)
     │
     │  Raw predictions
     ▼
YOLOv5 NMS (conf ≥ 0.5, iou ≥ 0.5)
     │
     │  Filtered detections [x1, y1, x2, y2, conf, class]
     ▼
JSON response → WebSocket → Browser
     │
     ▼
Canvas overlay with bounding boxes + labels
```

For image uploads, the same pipeline runs via a REST POST request to `/upload_frame`.

---


## 📦 Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework + WebSocket server |
| `uvicorn` | ASGI server |
| `onnxruntime` | Fast CPU inference for ONNX model |
| `torch` + `torchvision` | NMS post-processing via YOLOv5 utils |
| `yolov5` | `non_max_suppression` utility |
| `Pillow` | Image decoding and resizing |
| `numpy` | Array operations for model input |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

