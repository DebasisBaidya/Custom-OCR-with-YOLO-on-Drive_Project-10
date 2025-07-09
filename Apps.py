# ✅ Streamlit App for Project 10 - Custom OCR (YOLOv5 + Tesseract)
# Purpose: Detect and structure medical lab report fields using YOLOv5 and Tesseract OCR

import cv2
import numpy as np
import pytesseract as py
import pandas as pd
import streamlit as st
from PIL import Image
import os

# ✅ Load YOLOv5 model (your version)
def load_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.stop()
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ Run YOLOv5 on uploaded image
def predict_yolo(model, image):
    INPUT_WH_YOLO = 640
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_image

# ✅ Process YOLO predictions
def process_predictions(predictions, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes = []
    confidences = []
    detections = predictions[0]
    H, W = input_image.shape[:2]
    x_factor = W / 640
    y_factor = H / 640

    for det in detections:
        confidence = det[4]
        if confidence > conf_threshold:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_threshold:
                cx, cy, w, h = det[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                boxes.append([left, top, int(w * x_factor), int(h * y_factor)])
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes

# ✅ Preprocess cropped image for better OCR
def preprocess_image(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = cv2.bitwise_not(thresh)
    return roi

# ✅ Perform OCR and extract structured fields
def perform_ocr_on_crops(image, boxes, indices, ocr_config='--oem 3 --psm 6'):
    test_names, values, units, ref_ranges = [], [], [], []

    # Replace these indices with your actual class index mapping
    box_mapping = {
        61: "Test Name",
        14: "Value",
        26: "Units",
        41: "Reference Range"
    }

    for i in indices.flatten():
        if i >= len(boxes):
            continue

        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)
        x_end, y_end = min(image.shape[1], x+w), min(image.shape[0], y+h)
        w, h = x_end - x, y_end - y

        if w <= 0 or h <= 0:
            continue

        crop = image[y:y+h, x:x+w]
        roi = preprocess_image(crop)

        text = py.image_to_string(roi, config=ocr_config).strip()
        lines = text.splitlines()

        label = box_mapping.get(i, None)
        if label:
            for line in lines:
                if label == "Test Name": test_names.append(line.strip())
                elif label == "Value": values.append(line.strip())
                elif label == "Units": units.append(line.strip())
                elif label == "Reference Range": ref_ranges.append(line.strip())

    # Pad lists to equal length
    max_len = max(len(test_names), len(values), len(units), len(ref_ranges))
    test_names.extend([""] * (max_len - len(test_names)))
    values.extend([""] * (max_len - len(values)))
    units.extend([""] * (max_len - len(units)))
    ref_ranges.extend([""] * (max_len - len(ref_ranges)))

    return pd.DataFrame({
        "Test Name": test_names,
        "Value": values,
        "Units": units,
        "Reference Range": ref_ranges
    })

# ✅ Draw detection boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Streamlit UI
st.set_page_config(layout="wide")
st.title("🩺 Custom OCR for Medical Reports (YOLOv5 + Tesseract)")

uploaded_files = st.file_uploader("📤 Upload JPG image(s)", type="jpg", accept_multiple_files=True)

if uploaded_files:
    model = load_model()

    for uploaded_file in uploaded_files:
        st.markdown(f"### 📌 Processing: `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        preds, input_img = predict_yolo(model, image)
        indices, boxes = process_predictions(preds, input_img)

        if len(indices) == 0:
            st.warning("⚠️ No text regions detected!")
            continue

        # OCR + Draw
        df = perform_ocr_on_crops(image, boxes, indices)
        st.success("✅ OCR Complete!")

        st.dataframe(df.style.set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#f7f7f7"), ("color", "#333")]}
        ]).set_properties(**{"text-align": "left"}))

        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        boxed_img = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_img, caption="🔍 Detected Regions", use_column_width=True)
