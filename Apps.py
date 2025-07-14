import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

# Mapping detected class IDs to field names
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# Loading YOLOv5 ONNX model from file
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    return model

# Running YOLO prediction on the image
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# Processing predictions and applying Non-Maximum Suppression (NMS)
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
            class_id = int(np.argmax(scores))
            if scores[class_id] > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes, class_ids

# Extracting OCR text for each detected box and building the table without unit classification
def extract_table(image, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {key: [] for key in class_map.values()}

    for i in indices:
        if i >= len(boxes) or i >= len(class_ids):
            continue
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Converting crop to grayscale and resizing for better OCR accuracy
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            # Reading text lines from the ROI using EasyOCR
            lines = reader.readtext(roi, detail=0)
        except Exception:
            lines = []

        # Joining all OCR lines in this box as a single string preserving spaces
        text = " ".join([line.strip() for line in lines if line.strip()])
        results[label].append(text)

    # Padding columns to equal length for DataFrame construction
    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    df = pd.DataFrame(results)
    return df

# Drawing bounding boxes on the image with labels
def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x, y - 10 if y - 10 > 10 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return image

# Initializing Streamlit page configuration
st.set_page_config(page_title="Lab Report OCR - No Unit Classification", layout="centered", page_icon="🧾")

# Showing header
st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR - Extract Table Without Unit Classification</h2>", unsafe_allow_html=True)

# Showing upload instructions with a line break and no extra spacing
st.markdown(
    "<div style='text-align:center; font-size:16px;'>"
    "📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b><br>"
    "📂 Please upload lab report images to start extraction."
    "</div>",
    unsafe_allow_html=True,
)

# Showing file uploader below the instructions
uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Handling uploaded files
if uploaded_files:
    model = load_yolo_model()
    for file in uploaded_files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>", unsafe_allow_html=True)

        with st.spinner("🔍 Running YOLO Detection and OCR..."):
            image = np.array(Image.open(file).convert("RGB"))
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)
            if len(indices) == 0:
                st.warning("⚠️ No fields detected in this image.")
                continue
            df = extract_table(image, boxes, indices, class_ids)

        st.markdown("<h5 style='text-align:center;'>✅ Extraction Complete!</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(image.copy(), boxes, indices, class_ids), use_container_width=True)

        # Showing buttons for download and reset side-by-side aligned right
        c1, c2 = st.columns([3, 1])
        with c2:
            col_dl, col_rst = st.columns(2)
            with col_dl:
                # Providing CSV download button for extracted table
                st.download_button(
                    "⬇️ Download CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv"
                )
            with col_rst:
                # Showing reset button and clearing session state on click
                if st.button("🔄 Reset All"):
                    st.session_state.clear()
                    st.experimental_rerun()
