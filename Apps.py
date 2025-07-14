import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

# 🧠 Class Mapping
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# ✅ Load YOLOv5 ONNX Model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ YOLOv5 ONNX model not found!")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    return model

# 🔍 YOLOv5 Detection
def predict_yolo(model, image):
    h, w = image.shape[:2]
    input_img = np.zeros((max(h, w), max(h, w), 3), dtype=np.uint8)
    input_img[:h, :w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# 📦 Process YOLO Predictions
def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, scores, class_ids = [], [], []
    h, w = input_img.shape[:2]
    x_factor, y_factor = w / 640, h / 640
    for det in preds[0]:
        if det[4] > conf_thresh:
            cls_scores = det[5:]
            cls_id = np.argmax(cls_scores)
            if cls_scores[cls_id] > score_thresh:
                cx, cy, bw, bh = det[0:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                scores.append(float(det[4]))
                class_ids.append(cls_id)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes, class_ids

# 🔡 OCR with EasyOCR + Smart Mapping
def extract_fields_easyocr(image, boxes, indices, class_ids):
    reader = easyocr.Reader(['en'], gpu=False)
    results = {k: [] for k in class_map.values()}

    for i in indices:
        if i >= len(boxes) or i >= len(class_ids): continue
        x, y, w, h = boxes[i]
        crop = image[y:y+h, x:x+w]
        ocr_result = reader.readtext(crop, detail=0)
        text = " ".join(ocr_result).strip()
        label = class_map.get(class_ids[i], f"Class {class_ids[i]}")
        if text: results[label].append(text)

    # ✅ Smart Unit Correction
    auto_units = []
    for t in results["Reference Range"]:
        if '/' in t or 'IU' in t.upper() or 'ml' in t.lower() or 'g/' in t.lower():
            auto_units.append(t)
    results["Reference Range"] = [t for t in results["Reference Range"] if t not in auto_units]
    results["Units"].extend(auto_units)

    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        results[k].extend([""] * (max_len - len(results[k])))

    return pd.DataFrame(results)

# 🖼️ Draw Bounding Boxes
def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image

# 🎯 Streamlit UI
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")
st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> to test and upload from this: "
    "<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True
)

st.markdown("<div style='text-align:center;'>📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()

    for file in uploaded_files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>", unsafe_allow_html=True)

        with st.spinner(" "):
            st.markdown("<div style='text-align:center;'>🔍 Running YOLOv5 Detection and OCR...</div>", unsafe_allow_html=True)
            image = np.array(Image.open(file).convert("RGB"))
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected in this image.")
                continue

            df = extract_fields_easyocr(image, boxes, indices, class_ids)

        st.success("✅ Extraction Complete!")

        st.markdown("<h5 style='text-align:center;'>📊 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>🖼️ Annotated Image</h5>", unsafe_allow_html=True)
        annotated = draw_boxes(image.copy(), boxes, indices, class_ids)
        st.image(annotated, caption=f"📌 Detected Fields", use_container_width=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            col_dl, col_rst = st.columns(2)
            with col_dl:
                st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv")
            with col_rst:
                if st.button("🔄 Reset All"):
                    st.session_state.clear()
                    st.experimental_rerun()
