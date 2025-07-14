import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import easyocr
import os

# Class Map
class_map = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}

# Load YOLOv5 ONNX model
def load_model():
    model = cv2.dnn.readNetFromONNX("best.onnx")
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# YOLOv5 Detection
def predict(model, image):
    h, w, _ = image.shape
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square[:h, :w] = image
    blob = cv2.dnn.blobFromImage(square, 1/255.0, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, square

# Postprocess YOLO detections
def process(preds, image, conf_thresh=0.4, score_thresh=0.25):
    boxes, scores, class_ids = [], [], []
    h, w = image.shape[:2]
    x_scale, y_scale = w / 640, h / 640
    for det in preds[0]:
        if det[4] > conf_thresh:
            cls_scores = det[5:]
            cls_id = np.argmax(cls_scores)
            if cls_scores[cls_id] > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw/2) * x_scale)
                y = int((cy - bh/2) * y_scale)
                boxes.append([x, y, int(bw * x_scale), int(bh * y_scale)])
                scores.append(float(det[4]))
                class_ids.append(cls_id)
    idx = cv2.dnn.NMSBoxes(boxes, scores, score_thresh, 0.45)
    return idx.flatten() if len(idx) > 0 else [], boxes, class_ids

# OCR using EasyOCR
def extract_fields_easyocr(image, boxes, indices, class_ids):
    reader = easyocr.Reader(['en'])
    field_data = {k: [] for k in class_map.values()}

    for i in indices:
        if i >= len(boxes): continue
        x, y, w, h = boxes[i]
        x, y = max(x, 0), max(y, 0)
        crop = image[y:y+h, x:x+w]
        results = reader.readtext(crop)
        text = " ".join([res[1] for res in results]).strip()
        if not text: continue
        label = class_map.get(class_ids[i], f"Class {class_ids[i]}")
        field_data[label].append(text)

    # Smart units fix
    auto_units = []
    for val in field_data["Reference Range"]:
        if '/' in val or 'IU' in val.upper() or 'ml' in val.lower() or 'g/' in val.lower():
            auto_units.append(val)
    field_data["Reference Range"] = [v for v in field_data["Reference Range"] if v not in auto_units]
    field_data["Units"].extend(auto_units)

    # Equalize column lengths
    max_len = max([len(v) for v in field_data.values()] + [1])
    for k in field_data:
        field_data[k] += [""] * (max_len - len(field_data[k]))

    return pd.DataFrame(field_data)

# Draw Boxes
def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], f"Class {class_ids[i]}")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image

# UI Layout
st.set_page_config(page_title="Lab Report OCR", layout="centered")
st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> to test and upload from this: "
    "<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader("📤 Upload lab report images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Processing Block
if uploaded_files:
    model = load_model()
    for file in uploaded_files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>", unsafe_allow_html=True)
        
        # Centered spinner workaround
        with st.spinner(" "):
            st.markdown("""
                <div style='text-align:center; font-size:18px;'>
                    🔍 <b>Running YOLOv5 Detection and OCR...</b>
                </div>
                <style>
                .stSpinner > div > div {
                    margin: auto !important;
                }
                </style>
            """, unsafe_allow_html=True)

            image = np.array(Image.open(file).convert("RGB"))
            preds, yolo_img = predict(model, image)
            indices, boxes, class_ids = process(preds, yolo_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_fields_easyocr(image, boxes, indices, class_ids)

        st.success("✅ Extraction Complete!")
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Annotated Image</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(image.copy(), boxes, indices, class_ids), use_container_width=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv")
            with c2:
                if st.button("🔄 Reset All"):
                    st.session_state.clear()
                    st.experimental_rerun()
