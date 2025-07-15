# --------------------------------------------------
# 🏗️ I'm importing the required libraries
# --------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

# -------------------------------
# Class Mapping (Based on YOLOv5 class IDs)
# -------------------------------
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# -------------------------------
# Value + Unit Split Helper
# -------------------------------
import re
_unit_rx = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([^\d\s]+.*)?$", re.I)
UNIT_NORMALISE = {
    "g/dl": "g/dL", "mg/dl": "mg/dL", "mmol/l": "mmol/L", "μiu/ml": "µIU/mL",
    "iu/ml": "IU/mL", "ng/dl": "ng/dL", "ug/dl": "µg/dL", "μg/dl": "µg/dL"
}

def _split_value_unit(txt: str):
    m = _unit_rx.match(txt)
    if not m:
        return txt.strip(), ""
    val, unit = m.groups()
    unit = (unit or "").lower().strip()
    unit = UNIT_NORMALISE.get(unit, unit)
    return val.strip(), unit

# -------------------------------
# Load YOLOv5 ONNX Model
# -------------------------------
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("Model file not found: best.onnx")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# -------------------------------
# YOLO Inference
# -------------------------------
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# -------------------------------
# Process YOLO Outputs
# -------------------------------
def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    detections = preds[0]
    h, w = input_img.shape[:2]
    x_factor, y_factor = w / 640, h / 640
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
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten(), boxes, class_ids

# -------------------------------
# Perform OCR class-wise using EasyOCR
# -------------------------------
def perform_easyocr(image, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {label: [] for label in class_map.values()}

    for idx in indices:
        if idx >= len(boxes): continue
        x, y, w, h = boxes[idx]
        label = class_map.get(class_ids[idx], "Field")
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: continue

        # Preprocessing
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(th)

        try:
            lines = reader.readtext(roi, detail=0)
        except:
            lines = []

        for line in lines:
            clean = line.strip()
            if not clean:
                continue

            # Handle broken decimals like [2, .14]
            if label == "Value" and clean.startswith("."):
                if results["Value"]:
                    results["Value"][-1] += clean  # merge with previous
                    continue

            if label == "Value":
                val, unit = _split_value_unit(clean)
                results["Value"].append(val)
                if unit:
                    results["Units"].append(unit)
                continue

            results[label].append(clean)

    # Padding
    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))
    return pd.DataFrame(results)

# -------------------------------
# Draw Bounding Boxes
# -------------------------------
def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10 if y - 10 > 10 else y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown(
    "<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor (EasyOCR)</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports</b>: "
    "<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True,
)

st.markdown("""
<div style='text-align:center; margin-bottom:0;'>
📤 <b>Upload lab reports (.jpg, .jpeg, .png)</b><br>
<small>Upload one or more lab report images to extract structured data.</small>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    for file in uploaded_files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing: {file.name}</h4>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            with st.spinner("🔍 Running YOLOv5 Detection and EasyOCR..."):
                image = np.array(Image.open(file).convert("RGB"))
                preds, input_img = predict_yolo(model, image)
                indices, boxes, class_ids = process_predictions(preds, input_img)
                if len(indices) == 0:
                    st.warning("⚠️ No fields detected.")
                    continue
                df = perform_easyocr(image, boxes, indices, class_ids)

        st.markdown("<h5 style='text-align:center;'>✅ Extraction Complete</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Annotated Detection</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(image.copy(), boxes, indices, class_ids), use_column_width=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            col_dl, col_rst = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{file.name}_ocr.csv",
                    mime="text/csv",
                )
            with col_rst:
                if st.button("🧹 Clear All"):
                    st.session_state["uploaded_files"] = []
                    st.session_state["extracted_dfs"] = []
                    st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
                    st.rerun()
