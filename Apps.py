import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import easyocr

# ✅ I'm mapping YOLO class IDs to human-readable labels
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# ✅ I'm loading the ONNX YOLO model and ensuring it's using CPU backend
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ I'm resizing the input image to a square and performing YOLO inference
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ I'm filtering out predictions based on confidence and score thresholds
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

# ✅ I'm extracting text from each detected region using OCR
def extract_fields_exploded(image, boxes, indices, class_ids, reader):
    results = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    for i in indices:
        if i >= len(boxes) or i >= len(class_ids):
            continue

        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i])
        if not label:
            continue

        # I'm preparing each region before passing it to the OCR engine
        crop = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            ocr_lines = reader.readtext(roi, detail=0)
        except:
            ocr_lines = []

        for line in ocr_lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    # I'm converting the dictionary to a DataFrame for easier display
    df = pd.DataFrame({col: pd.Series(vals) for col, vals in results.items()})
    return df

# ✅ I'm merging consecutive Test Name fragments into a single row
def merge_fragmented_test_names(df):
    rows = df.to_dict("records")
    merged_rows = []
    buffer = None

    for row in rows:
        if row.get("Test Name") and not any([row.get("Value"), row.get("Units"), row.get("Reference Range")]):
            if buffer:
                buffer["Test Name"] += " " + row["Test Name"]
            else:
                buffer = row
        else:
            if buffer:
                merged_rows.append(buffer)
                buffer = None
            merged_rows.append(row)

    if buffer:
        merged_rows.append(buffer)

    return pd.DataFrame(merged_rows)

# ✅ I'm drawing green bounding boxes for detected fields
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ UI SECTION: Initial layout and toggles
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")
st.markdown("<h2 style='text-align: center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)

# Google Drive link to test files
st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> to test and upload from this: "
    "<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True
)

# Theme selection radio (centered)
st.markdown("<div style='text-align:center;'>🌗 <b>Choose Theme</b></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    theme = st.radio("", ["Light", "Dark"], index=0, horizontal=True, label_visibility="collapsed")

# OCR engine toggle (centered)
st.markdown("<div style='text-align:center;'>🧠 <b>Select OCR Engine</b></div>", unsafe_allow_html=True)
col4, col5, col6 = st.columns([1, 2, 1])
with col5:
    ocr_engine = st.radio("", ["EasyOCR", "Pytesseract"], index=0, horizontal=True, label_visibility="collapsed")

if ocr_engine == "Pytesseract":
    st.markdown("<div style='text-align:center; color:gray;'>⚠️ Must be installed at: <code>C:\\Program Files\\Tesseract-OCR\\tesseract.exe</code></div>", unsafe_allow_html=True)

# Instruction collapsible section
with st.expander("📘 How it works", expanded=False):
    st.markdown("""
    1. **Upload** `.jpg` lab reports.
    2. YOLOv5 detects fields: Test Name, Value, Units, Reference Range.
    3. EasyOCR or Pytesseract extracts text from those regions.
    4. Fragmented test names are auto-merged.
    5. You can download CSV + view overlay image.
    """)

# Upload message and uploader
st.markdown("<div style='text-align:center;'>📤 Upload JPG lab reports<br>📂 Please upload one or more JPG files to begin.</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type=["jpg"], accept_multiple_files=True)

# ✅ Processing uploaded files
if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"---\n### 📄 Processing File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        # Showing processing spinner
        with st.spinner("🔍 Running YOLOv5 Detection and OCR..."):
            st.markdown("<div style='text-align:center;'>🔍 Running YOLOv5 Detection and OCR...</div>", unsafe_allow_html=True)
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected in this image.")
                continue

            df = extract_fields_exploded(image, boxes, indices, class_ids, reader)
            df = merge_fragmented_test_names(df)

        st.success("✅ Extraction Complete!")

        # Showing results table
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        # Showing bounding box overlay
        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(image.copy(), boxes, indices), use_container_width=True)

        # CSV download + Reset button (centered)
        col7, col8, col9 = st.columns([1, 2, 1])
        with col8:
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv")
            if st.button("🔄 Reset All"):
                st.session_state.clear()
                st.experimental_rerun()
