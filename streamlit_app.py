import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
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

    original_h, original_w = frame.shape[:2]

    img = cv2.resize(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    outputs = onnx_session.run(
        [output_name],
        {input_name: img}
    )[0]

    predictions = outputs[0]

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

        score = obj_conf * class_scores[class_id]

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

        scores.append(float(score))
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

            x_scale = original_w / 640
            y_scale = original_h / 640

            x1 = int(x * x_scale)
            y1 = int(y * y_scale)

            x2 = int((x + w) * x_scale)
            y2 = int((y + h) * y_scale)

            detections.append({
                "box": [x1, y1, x2, y2],
                "conf": scores[idx],
                "class_id": class_ids[idx],
                "label": class_names[class_ids[idx]]
            })

    return detections

class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.last_inference_time = 0
        self.latest_result = []

        self.class_counter = defaultdict(int)
        self.logged_classes = set()

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        current_time = time.time()

        if current_time - self.last_inference_time >= 0.25:

            self.last_inference_time = current_time

            detections = detect_objects(img)
            self.latest_result = detections

            current_classes = {
                det["label"]
                for det in detections
            }

            for cls in class_names:

                if cls in current_classes:
                    self.class_counter[cls] += 1
                else:
                    self.class_counter[cls] = 0

                    if cls in self.logged_classes:
                        self.logged_classes.remove(cls)

        else:
            detections = self.latest_result

        for det in detections:

            label = det["label"]

            if self.class_counter[label] < THRESHOLD:
                continue

            if label not in self.logged_classes:
                self.logged_classes.add(label)

            x1, y1, x2, y2 = det["box"]

            color = CLASS_COLORS[label]

            text = (
                f"{label} "
                f"{det['conf']:.2f}"
            )

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

            font_color = (
                (0, 0, 0)
                if label == "smoke"
                else (255, 255, 255)
            )

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