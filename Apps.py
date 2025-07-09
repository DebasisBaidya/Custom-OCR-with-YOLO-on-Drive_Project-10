import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import easyocr

# ✅ Load YOLOv5 ONNX model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ Run YOLOv5 detection
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Process predictions (no class filtering)
def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences = [], []
    detections = preds[0]
    h, w = input_img.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for det in detections:
        conf = det[4]
        if conf > conf_thresh:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes

# ✅ OCR + explode logic from main.py
def perform_ocr_explode(image, boxes, indices, reader):
    box_mapping = {
        36: "Test Name",
        27: "Value",
        29: "Units",
        34: "Reference Range"
    }

    data = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    for i in indices:
        if i >= len(boxes):
            continue
        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(x + w, image.shape[1]), min(y + h, image.shape[0])
        crop = image[y:y2, x:x2]
        roi = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(roi)

        results = reader.readtext(roi, detail=0)
        lines = [line.strip() for line in results if line.strip()]
        if i in box_mapping:
            label = box_mapping[i]
            data[label].extend(lines)

    # Convert to exploded rows
    df = pd.DataFrame({col: pd.Series(vals) for col, vals in data.items()})
    return df

# ✅ Draw boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Streamlit App
st.set_page_config(layout="wide")
st.title("🧾 Medical OCR (main.py logic, EasyOCR, YOLOv5 ONNX)")

uploaded_files = st.file_uploader("📤 Upload JPG medical reports", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Detecting and extracting..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No boxes detected.")
                continue

            df = perform_ocr_explode(image, boxes, indices, reader)

        st.success("✅ Done!")
        st.dataframe(df)

        st.download_button("📥 Download CSV",
                           df.to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        st.image(draw_boxes(image.copy(), boxes, indices), caption="Detected Fields", use_container_width=True)
