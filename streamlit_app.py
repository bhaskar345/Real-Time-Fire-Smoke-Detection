import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import torch
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from yolov5.utils.general import non_max_suppression
from collections import defaultdict

class_counter = defaultdict(int)

THRESHOLD = 10

CLASS_COLORS = {
    "fire": (0, 0, 255),
    "smoke": (255, 255, 255),
    "other": (0, 255, 0)
}

onnx_session = ort.InferenceSession(
    "fire_n_smoke.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

class_names = ['fire', 'other', 'smoke']


def detect_objects(frame):
    """
    Run ONNX inference on frame
    """

    original_h, original_w = frame.shape[:2]

    img = cv2.resize(frame, (640, 640))

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outputs = onnx_session.run(
        [output_name],
        {input_name: img}
    )[0]

    preds = torch.tensor(outputs)

    preds = non_max_suppression(
        preds,
        conf_thres=0.5,
        iou_thres=0.5
    )

    detections = []

    if len(preds) > 0:
        for det in preds[0]:

            x1, y1, x2, y2, conf, cls = det[:6]

            x_scale = original_w / 640
            y_scale = original_h / 640

            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            detections.append({
                "box": [x1, y1, x2, y2],
                "conf": float(conf),
                "class_id": int(cls),
                "label": class_names[int(cls)]
            })
            print(class_names[int(cls)])

    return detections

class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.last_inference_time = 0
        self.latest_result = None

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        current_time = time.time()

        # Run inference only every 0.25 sec (4 FPS)
        if current_time - self.last_inference_time >= 0.25:

            self.last_inference_time = current_time

            detections = detect_objects(img)

            self.latest_result = detections

        else:
            detections = self.latest_result or []

        for det in detections:

            x1, y1, x2, y2 = det["box"]

            label = det["label"]
            color = CLASS_COLORS.get(label, (0, 255, 0))

            text = f"{label} {det['conf']:.2f}"

            (text_w, text_h), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.rectangle(
                img,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                color,
                -1
            )

            font_color = (0, 0, 0) if label == "smoke" else (255, 255, 255)

            cv2.putText(
                img,
                text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                font_color,
                2
            )

        return img


st.markdown("""
<style>
video {
    width: 640px !important;
    height: 480px !important;
    margin: auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Fire & Smoke Detection",
    layout="centered"
)

st.title("🔥 Fire & Smoke Detection")

st.write("Real-time ONNX YOLO Detection using Webcam")

webrtc_streamer(
    key="fire-detection",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)