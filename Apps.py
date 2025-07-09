# ✅ Streamlit App for Project 10 - Custom OCR
# Purpose: Detect regions from medical lab reports using YOLOv3 and extract text using Tesseract OCR

import cv2
import pytesseract as py
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image

# 📌 Task 1.1: Load YOLO model from repo
# Model is in 'best.onnx' relative to app.py
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please check the path or upload the model.")
        st.stop()
    try:
        model = cv2.dnn.readNet(model_path)  # More flexible than readNetFromONNX
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return model
    except cv2.error as e:
        st.error(f"❌ OpenCV failed to load the ONNX model.\n\nDetails: {e}")
        st.stop()

# 📌 Task 1.2: Perform YOLO prediction on uploaded image
@st.cache_resource
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

# 📌 Task 2.1: Extract useful bounding boxes based on confidence scores
def process_predictions(predictions, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes = []
    confidences = []
    detections = predictions[0]
    h, w = input_image.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for detection in detections:
        conf = detection[4]
        if conf > conf_threshold:
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            if class_score > score_threshold:
                cx, cy, bw, bh = detection[0:4]
                left = int((cx - 0.5 * bw) * x_factor)
                top = int((cy - 0.5 * bh) * y_factor)
                boxes.append([left, top, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes

# 📌 Task 2.2: Preprocess individual cropped image for OCR
def preprocess_image(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_not(thresh)

# 📌 Task 2.3: Perform OCR on all selected crop boxes
def perform_ocr_on_crops(image, boxes, indices, ocr_config='--oem 3 --psm 6'):
    results = []
    for i in indices.flatten():
        if i >= len(boxes):
            continue

        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)
        x_end = min(image.shape[1], x + w)
        y_end = min(image.shape[0], y + h)
        crop_img = image[y:y_end, x:x_end]
        roi = preprocess_image(crop_img)
        text = py.image_to_string(roi, config=ocr_config).strip()
        results.append(text)

    return pd.DataFrame({'Text': results})

# 📌 Task 3: Draw green bounding boxes for user visualization
def draw_bounding_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Task 4: Build Streamlit UI
st.set_page_config(layout="wide")
st.title("🩺 Custom OCR for Medical Lab Reports")

# 📌 Task 4.1: Upload multiple JPG images
uploaded_files = st.file_uploader("Upload JPG image(s)", type="jpg", accept_multiple_files=True)

# 📌 Task 4.2: Process each uploaded file
if uploaded_files:
    model = load_yolo_model()
    for uploaded_file in uploaded_files:
        st.markdown(f"### Processing: {uploaded_file.name}")
        image = np.array(Image.open(uploaded_file))

        # Step 1: Predict bounding boxes using YOLO
        preds, input_image = predict_yolo(model, image)
        indices, boxes = process_predictions(preds, input_image)

        # Step 2: Perform OCR on each cropped region
        ocr_df = perform_ocr_on_crops(image, boxes, indices)
        st.dataframe(ocr_df)

        # Step 3: Show download button (no Drive save)
        st.download_button(label="Download CSV", data=ocr_df.to_csv(index=False),
                           file_name=f'{uploaded_file.name}_ocr.csv', mime='text/csv')

        # Step 4: Show annotated image
        boxed_img = draw_bounding_boxes(image.copy(), boxes, indices)
        st.image(boxed_img, caption=f"Annotated Image for {uploaded_file.name}", use_column_width=True)
