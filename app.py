from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from yolov5.utils.general import non_max_suppression
import base64, json, io
from PIL import Image
import numpy as np
import onnxruntime as ort
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load ONNX model
onnx_session = ort.InferenceSession("fire_n_smoke.onnx", providers=["CPUExecutionProvider"])
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# Class labels
class_names = ['fire', 'other', 'smoke']

# Image preprocessing and inference
def detect_objects(base64_str):
    # Remove data URL prefix
    base64_str = base64_str.split(",")[-1]
    image_bytes = base64.b64decode(base64_str)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((640, 640))

    img = np.array(image).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(img, axis=0)   # Add batch dimension

    outputs = onnx_session.run([output_name], {input_name: input_tensor})[0]
    preds = torch.tensor(outputs)
    preds = non_max_suppression(preds, conf_thres=0.5, iou_thres=0.5)

    detections = []
    for det in preds[0]:
        x1, y1, x2, y2, conf, cls = det[:6]
        detections.append({
            "x1": int(x1.item()),
            "y1": int(y1.item()),
            "x2": int(x2.item()),
            "y2": int(y2.item()),
            "conf": float(conf.item()),
            "class_id": int(cls.item()),
            "label": class_names[int(cls.item())]
        })

    return detections

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            detections = detect_objects(data)
            await websocket.send_text(json.dumps({"detections": detections}))
        except Exception as e:
            print(f"WebSocket error: {e}")
            await websocket.close()
            break

@app.post("/upload_frame")
async def upload_frame(request: Request):
    data = await request.json()
    base64_str = data["image"]
    detections = detect_objects(base64_str)
    return {"detections": detections}

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
