import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
import easyocr

# ✅ Mapping YOLOv5 class index
class_map = {
    0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"
}

# ✅ Load YOLO ONNX model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ Run prediction
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Process prediction output
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

# ✅ OCR extraction
def extract_fields(image, boxes, indices, class_ids, ocr_engine):
    results = {key: [] for key in class_map.values()}
    for i in indices:
        if i >= len(boxes): continue
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

# ✅ Merge fragmented Test Names
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

# ✅ Draw boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ STREAMLIT UI Starts
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")
st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b>: <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>", unsafe_allow_html=True)

# ✅ OCR Engine Toggle with Style
if "ocr_engine" not in st.session_state:
    st.session_state.ocr_engine = "EasyOCR"

st.markdown("<div style='text-align:center;'><b>🧠 Select OCR Engine</b></div><br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1.5, 3, 1.5])
with col2:
    col_easy, col_py = st.columns(2)
    with col_easy:
        if st.button("EasyOCR", key="easy_btn", use_container_width=True):
            st.session_state.ocr_engine = "EasyOCR"
    with col_py:
        if st.button("Pytesseract", key="pytess_btn", use_container_width=True):
            st.session_state.ocr_engine = "Pytesseract"

# ✅ CSS Styling for Toggle Highlight
highlight_css = f"""
<style>
div[data-testid="column"] button {{
    border-radius: 8px;
    padding: 8px 12px;
    font-weight: bold;
    border: 1px solid #ccc;
    background-color: white;
    color: black;
}}

div[data-testid="column"] button:has(span:contains("{st.session_state.ocr_engine}")) {{
    background-color: #ffcccc;
    color: black;
    border: 2px solid #d33;
}}
</style>
"""
st.markdown(highlight_css, unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center;'>Selected OCR Engine: <span style='color:red; font-weight:bold;'>{st.session_state.ocr_engine}</span></div><br>", unsafe_allow_html=True)

if st.session_state.ocr_engine == "Pytesseract":
    st.markdown("<div style='text-align:center; color:gray;'>⚠️ Requires Tesseract installed at: <code>C:\\Program Files\\Tesseract-OCR\\tesseract.exe</code></div>", unsafe_allow_html=True)

# ✅ How It Works
with st.expander("📘 How it works"):
    st.markdown("""
    1. Upload `.jpg`, `.jpeg`, or `.png` lab reports.
    2. YOLOv5 detects fields: Test Name, Value, Units, Reference Range.
    3. OCR (EasyOCR / Pytesseract) extracts text from fields.
    4. Table and annotated image displayed.
    """)

# ✅ Upload UI
st.markdown("<div style='text-align:center;'>📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# ✅ Processing logic
if uploaded_files:
    model = load_yolo_model()
    for file in uploaded_files:
        st.markdown(f"---\n### 📄 Processing File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Running YOLOv5 Detection and OCR..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)
            if len(indices) == 0:
                st.warning("⚠️ No fields detected in this image.")
                continue
            df = extract_fields(image, boxes, indices, class_ids, st.session_state.ocr_engine)
            df = merge_fragmented_test_names(df)

        st.success("✅ Extraction Complete!")
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(image.copy(), boxes, indices), use_container_width=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv")
            if st.button("🔄 Reset All"):
                st.session_state.clear()
                st.experimental_rerun()
