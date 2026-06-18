from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
import base64, json, io
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2

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
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(img, axis=0)

    outputs = onnx_session.run(
        [output_name],
        {input_name: input_tensor}
    )[0]

    predictions = outputs[0]  # (25200, 8)

    boxes = []
    scores = []
    class_ids = []

    conf_threshold = 0.5

    for pred in predictions:

        obj_conf = pred[4]

        if obj_conf < conf_threshold:
            continue

        class_scores = pred[5:]
        class_id = np.argmax(class_scores)

        score = float(obj_conf * class_scores[class_id])

        if score < conf_threshold:
            continue

        xc, yc, w, h = pred[:4]

        x1 = xc - w / 2
        y1 = yc - h / 2

        boxes.append([
            int(x1),
            int(y1),
            int(w),
            int(h)
        ])

        scores.append(score)
        class_ids.append(int(class_id))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=conf_threshold,
        nms_threshold=0.5
    )

    detections = []

    if len(indices) > 0:

        for idx in indices.flatten():

            x, y, w, h = boxes[idx]

            detections.append({
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h),
                "conf": float(scores[idx]),
                "class_id": class_ids[idx],
                "label": class_names[class_ids[idx]]
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
