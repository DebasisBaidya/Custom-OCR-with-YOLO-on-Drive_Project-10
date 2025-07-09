# ✅ Streamlit App for Project 10 - Custom OCR (EasyOCR + YOLOv5)
# Purpose: Detect regions from medical lab reports using YOLOv5 and extract text using EasyOCR

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image
import easyocr

# ✅ Class names based on your data.yaml
CLASS_NAMES = ["Test Name", "Value", "Units", "Reference Range"]

# ✅ Load YOLOv5 ONNX model
@st.cache_resource
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ YOLO model not found. Please upload 'best.onnx' to the working directory.")
        st.stop()
    net = cv2.dnn.readNet(model_path)
    return net

# ✅ Load EasyOCR reader
@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['en'], gpu=False)

# ✅ Run inference on image using YOLOv5
def run_yolo_inference(image, model):
    INPUT_SIZE = 640
    h, w, _ = image.shape
    max_dim = max(h, w)
    padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    padded[:h, :w] = image

    blob = cv2.dnn.blobFromImage(padded, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward()
    return outputs[0], padded

# ✅ Process YOLO output predictions
def extract_boxes(preds, image_shape, conf_thresh=0.4, iou_thresh=0.5):
    boxes, scores, class_ids = [], [], []
    h, w = image_shape[:2]
    x_factor, y_factor = w / 640, h / 640

    for det in preds:
        conf = det[4]
        if conf < conf_thresh:
            continue
        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        if class_scores[class_id] > 0.25:
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * x_factor)
            y = int((cy - bh / 2) * y_factor)
            width = int(bw * x_factor)
            height = int(bh * y_factor)
            boxes.append([x, y, width, height])
            scores.append(float(conf))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    final = [(boxes[i[0]], class_ids[i[0]]) for i in indices]
    return final

# ✅ Extract text using EasyOCR from each region and map to label
def extract_text_by_label(image, detections, reader):
    data = {"Test Name": "", "Value": "", "Units": "", "Reference Range": ""}
    for (x, y, w, h), class_id in detections:
        crop = image[y:y+h, x:x+w]
        if crop.size == 0: continue
        text = reader.readtext(crop, detail=0)
        label = CLASS_NAMES[class_id]
        if label in data:
            data[label] = " ".join(text).strip()
    return data

# ✅ Draw bounding boxes with class labels
def draw_boxes(image, detections):
    for (x, y, w, h), class_id in detections:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, CLASS_NAMES[class_id], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return image

# ✅ Streamlit App UI
st.set_page_config(layout="wide", page_title="🩺 Medical OCR App")
st.title("🩺 Custom OCR for Medical Lab Reports (YOLOv5 + EasyOCR)")

uploaded_files = st.file_uploader("📤 Upload JPG image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = load_easyocr_reader()
    all_data = []

    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### 🖼 Processing: `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file))

        with st.spinner("🔄 Running detection and OCR..."):
            preds, padded = run_yolo_inference(image, model)
            detections = extract_boxes(preds, padded)
            row_data = extract_text_by_label(image, detections, reader)
            all_data.append(row_data)
            boxed_image = draw_boxes(image.copy(), detections)

        st.image(boxed_image, caption="🔍 Detected Fields", use_container_width=True)

    df = pd.DataFrame(all_data)
    st.markdown("### 📋 Extracted Results")
    st.dataframe(df)

    st.download_button("📥 Download as CSV", data=df.to_csv(index=False), file_name="lab_ocr_output.csv", mime="text/csv")
