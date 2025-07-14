import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import easyocr

# I am mapping class IDs to meaningful labels for OCR fields
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# I am loading the ONNX YOLOv5 model from local path
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# I am preparing the image and passing it to the YOLO model for prediction
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# I am filtering model predictions and performing Non-Max Suppression
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

# I am extracting OCR text from each bounding box and organizing by label
def extract_fields_exploded(image, boxes, indices, class_ids, reader):
    results = {key: [] for key in class_map.values()}

    for i in indices:
        if i >= len(boxes) or i >= len(class_ids):
            continue

        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i])
        if not label:
            continue

        # I am preprocessing the cropped image for better OCR accuracy
        crop = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        # I am using EasyOCR to read text
        try:
            ocr_lines = reader.readtext(roi, detail=0)
        except:
            ocr_lines = []

        for line in ocr_lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    # I am converting the dictionary to a DataFrame
    df = pd.DataFrame({col: pd.Series(vals) for col, vals in results.items()})
    return df

# I am merging rows where test names are split across lines
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

# I am drawing YOLO bounding boxes on the image
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ──────────────────────────────────────────────
# ✅ Streamlit UI starts here
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

# I am creating top bar with theme toggle and reset button
top_left, top_right = st.columns([6, 1])
with top_right:
    theme = st.radio("", ["🌞", "🌙"], horizontal=True, label_visibility="collapsed")
    if st.button("🔄 Reset All"):
        st.session_state.clear()
        st.experimental_rerun()

# Title and description
st.markdown("<h2 style='text-align: center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your JPG lab reports to extract test names, values, units, and reference ranges using YOLOv5 and EasyOCR.</p>", unsafe_allow_html=True)

# I am linking to Google Drive for sample JPGs
st.markdown("<p style='text-align: center;'>📥 Download sample Lab Reports (JPG) to test and upload from this: <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></p>", unsafe_allow_html=True)

# I am displaying help section
with st.expander("📘 How it works", expanded=False):
    st.markdown("""
    1. Upload one or more `.jpg` lab reports.  
    2. YOLO detects fields like Test Name, Value, Units, Reference Range.  
    3. EasyOCR extracts text from detected regions.  
    4. Smart merge logic combines split test names.  
    5. Final output shown in a table with CSV and image overlay.
    """)

# I am handling file upload and display message
st.markdown("<div style='text-align:center;'>📤 <b>Upload JPG lab reports</b><br><span style='color:gray;'>📂 Please upload one or more JPG files to begin.</span></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("", type=["jpg"], accept_multiple_files=True)

# I am processing uploaded files
if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"<hr><h4 style='text-align:center;'>📄 File: {file.name}</h4>", unsafe_allow_html=True)
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Running YOLOv5 Detection and OCR..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected in this image.")
                continue

            df = extract_fields_exploded(image, boxes, indices, class_ids, reader)
            df = merge_fragmented_test_names(df)

        st.success("✅ Extraction Complete!")

        # I am showing output table and download option
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
        with col2:
            st.markdown("<h5 style='text-align:center;'>📥 Download</h5>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download CSV",
                data=df.to_csv(index=False),
                file_name=f"{file.name}_ocr.csv",
                mime="text/csv"
            )

        # I am displaying YOLO box overlay on image
        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        boxed_image = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_image, use_container_width=True, caption="Detected Regions")
