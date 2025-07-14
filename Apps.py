import os
import cv2
import numpy as np
import pytesseract as py
import pandas as pd
import streamlit as st
from PIL import Image

# Class Mapping (Update if class ids differ)
box_mapping = {
    61: "Test Name",
    14: "Value",
    26: "Units",
    41: "Reference Range"
}

def load_yolo_model():
    model = cv2.dnn.readNetFromONNX('best.onnx')
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

def predict_yolo(model, image):
    wh = 640
    h, w = image.shape[:2]
    padded = np.zeros((max(h, w), max(h, w), 3), dtype=np.uint8)
    padded[:h, :w] = image
    blob = cv2.dnn.blobFromImage(padded, 1/255, (wh, wh), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, padded

def process_predictions(preds, image, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    h, w = image.shape[:2]
    x_factor, y_factor = w / 640, h / 640
    for det in preds[0]:
        if det[4] > conf_thresh:
            scores = det[5:]
            cls_id = int(np.argmax(scores))
            if scores[cls_id] > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw/2) * x_factor)
                y = int((cy - bh/2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(det[4]))
                class_ids.append(cls_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten(), boxes, class_ids

def preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_not(th)

def extract_table(image, boxes, indices, class_ids):
    data = {v: [] for v in box_mapping.values()}
    for i in indices:
        if i >= len(boxes) or i >= len(class_ids): continue
        x, y, w, h = boxes[i]
        crop = image[y:y+h, x:x+w]
        roi = preprocess(crop)
        text = py.image_to_string(roi, config='--oem 3 --psm 6').strip()
        lines = [t.strip() for t in text.splitlines() if t.strip()]
        label = box_mapping.get(class_ids[i])
        if label and lines:
            data[label].extend(lines)

    # Smart correction
    smart_units = [t for t in data["Reference Range"] if "/" in t or "IU" in t or "ml" in t or "g/" in t]
    data["Reference Range"] = [t for t in data["Reference Range"] if t not in smart_units]
    data["Units"].extend(smart_units)

    # Normalize
    max_len = max(len(v) for v in data.values())
    for k in data:
        data[k] += [""] * (max_len - len(data[k]))

    return pd.DataFrame(data)

def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = box_mapping.get(class_ids[i], f"Class {class_ids[i]}")
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return image

# ──────────────────────────────────────────────────────────────
# 💡 Streamlit Interface Starts Here
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")
st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> to test and upload from this: "
    "<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True
)

st.markdown("<div style='text-align:center;'>📤 <b>Upload lab reports (.jpg, .jpeg, .png)</b></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    for file in uploaded_files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>", unsafe_allow_html=True)
        with st.spinner("🔍 <div style='text-align:center;'>Running YOLOv5 Detection and OCR...</div>"):
            image = np.array(Image.open(file).convert("RGB"))
            preds, yolo_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, yolo_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_table(image, boxes, indices, class_ids)

        st.markdown("<div style='text-align:center;'>✅ Extraction Complete!</div>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(image.copy(), boxes, indices, class_ids), use_column_width=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            col_dl, col_rst = st.columns(2)
            with col_dl:
                st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv")
            with col_rst:
                if st.button("🔄 Reset All"):
                    st.session_state.clear()
                    st.rerun()
