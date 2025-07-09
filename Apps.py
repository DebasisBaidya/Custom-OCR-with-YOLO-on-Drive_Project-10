# ✅ Streamlit App for Project 10 - Custom OCR
# Purpose: Detect regions from medical lab reports using YOLOv3 and extract text using EasyOCR (no Tesseract dependency)

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image
import easyocr

# ✅ Load YOLOv3 model (ONNX format)
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please upload the ONNX model.")
        st.stop()
    try:
        model = cv2.dnn.readNet(model_path)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return model
    except cv2.error as e:
        st.error(f"❌ Failed to load model.\n\nDetails: {e}")
        st.stop()

# ✅ Perform YOLO object detection
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

# ✅ Extract bounding boxes from predictions
def process_predictions(predictions, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes, confidences, class_ids = [], [], []
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
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes, class_ids

# ✅ Preprocess and apply OCR with EasyOCR
def perform_easyocr_on_crops(image, boxes, indices, class_ids):
    reader = easyocr.Reader(['en'], gpu=False)
    class_names = ["Test Name", "Value", "Units", "Reference Range"]
    results = {label: [] for label in class_names}

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)
        x_end = min(image.shape[1], x + w)
        y_end = min(image.shape[0], y + h)
        crop_img = image[y:y_end, x:x_end]

        # Read text using EasyOCR
        text = reader.readtext(crop_img, detail=0, paragraph=False)
        label = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "Unknown"
        results[label].append(" ".join(text))

    # Make all lists same length
    max_len = max(len(v) for v in results.values())
    for key in results:
        while len(results[key]) < max_len:
            results[key].append("")

    return pd.DataFrame(results)

# ✅ Draw boxes on image for visual verification
def draw_boxes(image, boxes, indices, class_ids):
    class_names = ["Test Name", "Value", "Units", "Reference Range"]
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "?"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image

# ✅ Build Streamlit UI
st.set_page_config(layout="wide")
st.title("🩺 Custom OCR for Medical Lab Reports")
st.markdown("Upload a JPG lab report image to extract results in structured format.")

uploaded_files = st.file_uploader("Upload JPG image(s)", type="jpg", accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    for uploaded_file in uploaded_files:
        st.markdown(f"### Processing: {uploaded_file.name}")
        image = np.array(Image.open(uploaded_file))

        with st.spinner("🔍 Detecting and reading text from image..."):
            preds, input_image = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_image)
            ocr_df = perform_easyocr_on_crops(image, boxes, indices, class_ids)

        st.success("✅ Extraction complete!")

        st.dataframe(ocr_df, use_container_width=True)

        st.download_button(
            label="Download CSV",
            data=ocr_df.to_csv(index=False),
            file_name=f"{uploaded_file.name}_ocr.csv",
            mime="text/csv"
        )

        boxed_img = draw_boxes(image.copy(), boxes, indices, class_ids)
        st.image(boxed_img, caption="Detected Regions", use_container_width=True)
