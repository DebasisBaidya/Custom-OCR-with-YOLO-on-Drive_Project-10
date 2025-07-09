# ✅ Streamlit App for Project 10 - Custom OCR with YOLOv5 and EasyOCR
# Detects text boxes and extracts text from lab reports into structured format.

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image
import easyocr

# Define class labels in the correct column order
CLASS_NAMES = ["Test Name", "Value", "Units", "Reference Range"]

# ✅ Load YOLOv5 ONNX model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please upload the model.")
        st.stop()
    try:
        model = cv2.dnn.readNet(model_path)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# ✅ Predict with YOLOv5 ONNX
@st.cache_resource
def predict_yolo(_model, image):
    INPUT_WH = 640
    h, w = image.shape[:2]
    max_dim = max(h, w)
    padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    padded[:h, :w] = image
    blob = cv2.dnn.blobFromImage(padded, 1/255.0, (INPUT_WH, INPUT_WH), swapRB=True, crop=False)
    _model.setInput(blob)
    preds = _model.forward()
    return preds[0], padded

# ✅ Extract bounding boxes and class IDs
def extract_boxes(preds, image_shape, conf_thresh=0.4, iou_thresh=0.5):
    boxes, scores, class_ids = [], [], []
    h, w = image_shape[:2]
    x_factor, y_factor = w / 640, h / 640

    for det in preds:
        if len(det) < 6:
            continue
        conf = float(det[4])
        if conf < conf_thresh:
            continue
        class_scores = det[5:]
        if class_scores.size == 0:
            continue
        class_id = np.argmax(class_scores)
        if class_scores[class_id] > 0.25:
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * x_factor)
            y = int((cy - bh / 2) * y_factor)
            width = int(bw * x_factor)
            height = int(bh * y_factor)
            boxes.append([x, y, width, height])
            scores.append(conf)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    final = [(boxes[i], class_ids[i]) for i in indices.flatten()] if len(indices) > 0 else []
    return final

# ✅ Draw bounding boxes on image
def draw_boxes(image, results):
    for (box, class_id) in results:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"ID:{class_id}"
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# ✅ Run OCR on each box and align into structured rows
def extract_table_data(image, results, reader):
    row_dict = {}
    for (box, class_id) in results:
        x, y, w, h = box
        crop = image[y:y+h, x:x+w]
        text = " ".join(reader.readtext(crop, detail=0)).strip()
        if class_id in row_dict:
            row_dict[class_id].append((y, text))  # Handle duplicates by position
        else:
            row_dict[class_id] = [(y, text)]

    # Sort entries by vertical position
    rows = []
    max_rows = max([len(v) for v in row_dict.values()] + [0])
    for i in range(max_rows):
        row = []
        for class_id in range(len(CLASS_NAMES)):
            items = sorted(row_dict.get(class_id, []), key=lambda tup: tup[0])
            row.append(items[i][1] if i < len(items) else "")
        rows.append(row)

    return pd.DataFrame(rows, columns=CLASS_NAMES)

# ✅ Streamlit Interface
st.set_page_config(layout="wide")
st.title("🧪 Custom OCR for Medical Lab Reports")
uploaded_files = st.file_uploader("📤 Upload JPG image(s)", type="jpg", accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for uploaded_file in uploaded_files:
        st.markdown(f"### Processing: `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file))

        with st.spinner("🔍 Detecting regions and reading text..."):
            preds, padded = predict_yolo(model, image)
            results = extract_boxes(preds, padded)
            ocr_df = extract_table_data(padded, results, reader)
            annotated = draw_boxes(image.copy(), results)

        st.image(annotated, caption="📸 Annotated Image", use_container_width=True)
        st.dataframe(ocr_df)

        st.download_button(
            label="⬇️ Download OCR Results as CSV",
            data=ocr_df.to_csv(index=False),
            file_name=f"{uploaded_file.name}_ocr.csv",
            mime="text/csv"
        )
