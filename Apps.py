import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
import easyocr

# Mapping class IDs to field labels
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# Load YOLOv5 ONNX model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    return cv2.dnn.readNetFromONNX(model_path)

# Run YOLO prediction
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# Process YOLO predictions
def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
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
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes, class_ids

# OCR extraction
def extract_fields(image, boxes, indices, class_ids, ocr_engine):
    results = {key: [] for key in class_map.values()}
    for i in indices:
        if i >= len(boxes) or i >= len(class_ids):
            continue

        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i])
        if not label:
            continue

        crop = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            if ocr_engine == "EasyOCR":
                reader = easyocr.Reader(['en'], gpu=False)
                lines = reader.readtext(roi, detail=0)
            else:
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                lines = pytesseract.image_to_string(roi).splitlines()
        except:
            lines = []

        for line in lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    return pd.DataFrame({col: pd.Series(vals) for col, vals in results.items()})

# Merge test name fragments
def merge_fragmented_test_names(df):
    rows = df.to_dict("records")
    merged_rows, buffer = [], None
    for row in rows:
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

# Draw bounding boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# Streamlit config
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

# Main title
st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload JPG reports to extract medical test data using YOLOv5 + OCR</p>", unsafe_allow_html=True)

# Sample link
st.markdown("""
<div style='text-align:center;'>
📥 Download sample Lab Reports (JPG) to test and upload from this: 
<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a>
</div>
""", unsafe_allow_html=True)

# Center-aligned theme toggle
st.markdown("<div style='text-align:center; margin-top:10px;'>🌗 <b>Choose Theme</b></div>", unsafe_allow_html=True)
st.radio("", ["Light", "Dark"], index=0, horizontal=True, label_visibility="collapsed")

# How it works section
with st.expander("📘 How it works", expanded=False):
    st.markdown("""
    1. Upload `.jpg` lab reports  
    2. YOLO detects Test Name, Value, Units, Reference Range  
    3. OCR reads text using EasyOCR or Pytesseract  
    4. Merged rows are structured and exported  
    """)

# OCR selection (centered)
st.markdown("<div style='text-align:center;'>🧠 <b>Select OCR Engine</b></div>", unsafe_allow_html=True)
ocr_engine = st.selectbox("", ["EasyOCR", "Pytesseract"], index=0)
if ocr_engine == "Pytesseract":
    st.markdown(
        "<div style='text-align:center; color:gray;'>⚠️ Tesseract must be installed at: <code>C:\\Program Files\\Tesseract-OCR\\tesseract.exe</code></div>",
        unsafe_allow_html=True
    )

# Upload section
st.markdown("<div style='text-align:center;'>📤 <b>Upload JPG lab reports</b><br><span style='color:gray;'>📂 Please upload one or more JPG files to begin.</span></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("", type=["jpg"], accept_multiple_files=True)

# Process uploads
if uploaded_files:
    model = load_yolo_model()

    for file in uploaded_files:
        st.markdown(f"<hr><h4 style='text-align:center;'>📄 File: {file.name}</h4>", unsafe_allow_html=True)
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner(""):
            st.markdown("<div style='text-align:center;'>🔍 Running YOLOv5 Detection and OCR...</div>", unsafe_allow_html=True)
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_fields(image, boxes, indices, class_ids, ocr_engine)
            df = merge_fragmented_test_names(df)

        st.success("✅ Extraction Complete!")
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        boxed_img = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_img, use_container_width=True, caption="Detected Regions")

        # Final download + reset buttons centered
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download as CSV",
            data=df.to_csv(index=False),
            file_name=f"{file.name}_ocr.csv",
            mime="text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# Reset All button at bottom, center
st.markdown("<div style='text-align:center; margin-top: 20px;'>", unsafe_allow_html=True)
if st.button("🔄 Reset All"):
    st.session_state.clear()
    st.experimental_rerun()
st.markdown("</div>", unsafe_allow_html=True)
