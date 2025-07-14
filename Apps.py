import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
import easyocr

# 👉 I am mapping YOLO class indices to meaningful labels
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# 👉 I am loading the YOLOv5 ONNX model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# 👉 I am running YOLOv5 prediction on the input image
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# 👉 I am filtering predictions using thresholds and NMS
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

# 👉 I am extracting OCR text from detected boxes using selected OCR engine
def extract_fields(image, boxes, indices, class_ids, ocr_engine):
    results = {key: [] for key in class_map.values()}
    for i in indices:
        if i >= len(boxes) or i >= len(class_ids): continue
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i])
        if not label: continue
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
                pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
                lines = pytesseract.image_to_string(roi).splitlines()
        except:
            lines = []

        for line in lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    return pd.DataFrame({col: pd.Series(vals) for col, vals in results.items()})

# 👉 I am merging rows where only "Test Name" is present and others are missing
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

# 👉 I am drawing bounding boxes on image
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# 👉 I am configuring the Streamlit page
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

# 👉 I am showing the app title and download link
st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b>: "
    "<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True
)

# 👉 I am setting default OCR engine to EasyOCR
if "ocr_engine" not in st.session_state:
    st.session_state.ocr_engine = "EasyOCR"

# 👉 I am creating OCR engine selection section with styled buttons
st.markdown("<div style='text-align:center;'><b>🧠 Select OCR Engine</b></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    b1, b2 = st.columns(2)
    with b1:
        if st.button("✅ EasyOCR" if st.session_state.ocr_engine == "EasyOCR" else "EasyOCR"):
            st.session_state.ocr_engine = "EasyOCR"
    with b2:
        if st.button("✅ Pytesseract" if st.session_state.ocr_engine == "Pytesseract" else "Pytesseract"):
            st.session_state.ocr_engine = "Pytesseract"

# 👉 I am showing selected OCR engine inline with color highlight
st.markdown(
    f"<div style='text-align:center;'>Selected OCR Engine: <span style='color:crimson; font-weight:bold;'>{st.session_state.ocr_engine}</span></div>",
    unsafe_allow_html=True
)

# 👉 I am warning users if Pytesseract is selected
if st.session_state.ocr_engine == "Pytesseract":
    st.markdown("<div style='text-align:center; color:gray;'>⚠️ Make sure Tesseract is installed at: <code>C:\\Program Files\\Tesseract-OCR\\tesseract.exe</code></div>", unsafe_allow_html=True)

# 👉 I am showing help section
with st.expander("📘 How it works"):
    st.markdown("""
    1. Upload `.jpg`, `.jpeg`, or `.png` lab report images.
    2. YOLOv5 detects fields: Test Name, Value, Units, Reference Range.
    3. OCR extracts text using EasyOCR or Pytesseract.
    4. Merging is done for fragmented test names.
    5. Results are shown in a table and downloadable as CSV.
    """)

# 👉 I am placing the file uploader
st.markdown("<div style='text-align:center;'>📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 👉 I am processing files when uploaded
if uploaded_files:
    model = load_yolo_model()

    for file in uploaded_files:
        st.markdown("---")
        st.markdown(f"<h5>📄 Processing File: <code>{file.name}</code></h5>", unsafe_allow_html=True)
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Running YOLOv5 Detection and OCR..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected in this image.")
                continue

            df = extract_fields(image, boxes, indices, class_ids, st.session_state.ocr_engine)
            df = merge_fragmented_test_names(df)

        # 👉 I am displaying results
        st.success("✅ Extraction Complete!")
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(image.copy(), boxes, indices), use_container_width=True)

        # 👉 I am showing download and reset buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv")
            if st.button("🔄 Reset All"):
                st.session_state.clear()
                st.experimental_rerun()
