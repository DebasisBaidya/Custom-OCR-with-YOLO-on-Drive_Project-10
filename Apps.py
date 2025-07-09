# ✅ Streamlit App for Project 10 - Custom OCR (YOLOv5 + EasyOCR)

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import easyocr
import yaml

# ✅ Set Streamlit layout
st.set_page_config(layout="wide")
st.title("🩺 Custom OCR for Medical Lab Reports")

# ✅ Load class names from data.yaml
def load_class_names(yaml_path="data.yaml"):
    if not os.path.exists(yaml_path):
        st.error(f"❌ data.yaml not found at {yaml_path}")
        st.stop()
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get("names", ["Text"])

class_names = load_class_names()

# ✅ Load YOLOv5 ONNX model
def load_yolo_model(model_path="best.onnx"):
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.stop()
    try:
        model = cv2.dnn.readNet(model_path)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return model
    except cv2.error as e:
        st.error(f"❌ Failed to load ONNX model.\n\n{e}")
        st.stop()

# ✅ Perform YOLO inference
def predict(model, image):
    input_img = cv2.resize(image, (640, 640))
    blob = cv2.dnn.blobFromImage(input_img, 1/255.0, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds[0], input_img

# ✅ Post-process predictions and extract boxes
def extract_boxes(preds, original_image, conf_thresh=0.4, iou_thresh=0.5):
    h, w = original_image.shape[:2]
    x_factor, y_factor = w / 640, h / 640
    boxes, scores, class_ids = [], [], []

    for det in preds:
        if len(det) < 6:
            continue
        conf = float(det[4])
        if conf < conf_thresh:
            continue
        class_scores = det[5:]
        if len(class_scores) == 0:
            continue
        class_id = np.argmax(class_scores)
        if class_scores[class_id] < 0.25:
            continue

        try:
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * x_factor)
            y = int((cy - bh / 2) * y_factor)
            width = int(bw * x_factor)
            height = int(bh * y_factor)
            boxes.append([x, y, width, height])
            scores.append(conf)
            class_ids.append(class_id)
        except:
            continue

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    final = [(boxes[i], class_ids[i]) for i in indices.flatten()] if len(indices) > 0 else []
    return final

# ✅ EasyOCR reader initialization
@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['en'], gpu=False)

# ✅ Run OCR on cropped regions
def run_ocr(image, detections, reader):
    rows = []
    for box, class_id in detections:
        x, y, w, h = box
        crop = image[y:y+h, x:x+w]
        result = reader.readtext(crop, detail=0)
        text = " ".join(result).strip()
        label = class_names[class_id] if class_id < len(class_names) else "Unknown"
        rows.append((label, text))
    return rows

# ✅ Format results into 4-column DataFrame
def format_results(ocr_rows):
    grouped = {name: [] for name in class_names}
    for label, text in ocr_rows:
        if label in grouped:
            grouped[label].append(text)
    max_len = max([len(v) for v in grouped.values()])
    for k in grouped:
        while len(grouped[k]) < max_len:
            grouped[k].append("")
    return pd.DataFrame(grouped)

# ✅ Draw bounding boxes
def draw_boxes(image, detections):
    for box, class_id in detections:
        x, y, w, h = box
        label = class_names[class_id] if class_id < len(class_names) else "Unknown"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# ✅ File uploader
uploaded_files = st.file_uploader("📤 Upload JPG Image(s)", type="jpg", accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = load_easyocr_reader()

    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### 📄 File: `{uploaded_file.name}`")

        image = np.array(Image.open(uploaded_file).convert("RGB"))

        with st.spinner("🔍 Processing..."):
            preds, padded = predict(model, image)
            detections = extract_boxes(preds, image)
            ocr_data = run_ocr(image, detections, reader)
            df = format_results(ocr_data)
            boxed = draw_boxes(image.copy(), detections)

        st.image(boxed, caption="🔲 Detected Regions", use_container_width=True)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")
