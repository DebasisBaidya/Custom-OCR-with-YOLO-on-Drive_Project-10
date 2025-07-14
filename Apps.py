import os
import cv2
import numpy as np
import pandas as pd
import pytesseract as py
import streamlit as st
from PIL import Image
import easyocr

# ✅ Class map (YOLOv5 output classes)
class_map = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}

# ✅ Load YOLOv5 ONNX Model
def load_model(model_path="best.onnx"):
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# 🔍 Run YOLOv5 Detection
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square[:h, :w] = image
    blob = cv2.dnn.blobFromImage(square, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, square

# 📦 Post-process YOLO Detections
def process_predictions(preds, image, conf=0.4, score=0.25):
    boxes, scores, class_ids = [], [], []
    h, w = image.shape[:2]
    x_scale, y_scale = w / 640, h / 640
    for det in preds[0]:
        if det[4] > conf:
            cls_scores = det[5:]
            cls_id = int(np.argmax(cls_scores))
            if cls_scores[cls_id] > score:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw/2) * x_scale)
                y = int((cy - bh/2) * y_scale)
                boxes.append([x, y, int(bw * x_scale), int(bh * y_scale)])
                scores.append(float(det[4]))
                class_ids.append(cls_id)
    idx = cv2.dnn.NMSBoxes(boxes, scores, score, 0.45)
    return idx.flatten(), boxes, class_ids

# ✍️ Perform EasyOCR and postprocessing
def extract_fields(image, boxes, indices, class_ids):
    results = {v: [] for v in class_map.values()}
    reader = easyocr.Reader(['en'], gpu=False)

    for i in indices:
        if i >= len(boxes): continue
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i])
        crop = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(th)

        try:
            lines = reader.readtext(roi, detail=0)
        except:
            lines = []

        for line in lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    # 🔁 Smart Units Correction
    smart_units = []
    for t in results["Reference Range"]:
        if '/' in t or "IU" in t.upper() or "ml" in t.lower() or 'g/' in t.lower():
            smart_units.append(t)
    results["Reference Range"] = [t for t in results["Reference Range"] if t not in smart_units]
    results["Units"].extend(smart_units)

    # 🔗 Merge fragmented Test Names
    merged_rows, buffer = [], None
    for row in pd.DataFrame(results).to_dict("records"):
        if row.get("Test Name") and not any([row.get("Value"), row.get("Units"), row.get("Reference Range")]):
            buffer = row if not buffer else {"Test Name": buffer["Test Name"] + " " + row["Test Name"]}
        else:
            if buffer:
                merged_rows.append(buffer)
                buffer = None
            merged_rows.append(row)
    if buffer:
        merged_rows.append(buffer)

    return pd.DataFrame(merged_rows)

# 🖼️ Draw bounding boxes with class names
def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i])
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return image

# 🚀 Streamlit App UI
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")
st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> to test and upload from this:</div>"
    "<div style='text-align:center;'><a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True
)

st.markdown("<div style='text-align:center;'>📤 <b>Upload lab reports (.jpg, .jpeg, or .png)</b></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    for file in uploaded_files:
        st.markdown(f"<div style='text-align:center;'>📄 Processing File: <b>{file.name}</b></div>", unsafe_allow_html=True)
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Running YOLOv5 Detection and OCR..."):
            preds, yolo_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, yolo_img)
            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue
            df = extract_fields(image, boxes, indices, class_ids)

        # 🧾 Output Display
        st.success("✅ Extraction Complete!")
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Annotated Image</h5>", unsafe_allow_html=True)
        annotated = draw_boxes(image.copy(), boxes, indices, class_ids)
        st.image(annotated, use_container_width=True)

        # Download + Reset buttons centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv")
            if st.button("🔄 Reset All"):
                st.session_state.clear()
                st.experimental_rerun()
