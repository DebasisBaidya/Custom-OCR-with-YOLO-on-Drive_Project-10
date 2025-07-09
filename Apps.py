# ✅ Streamlit App for Custom OCR using YOLOv3 + EasyOCR
# Purpose: Detect fields in lab reports and extract their text
# Assumes model is trained on 4 classes and class names are in data.yaml

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import yaml
import easyocr

# ✅ Load class names from data.yaml
def load_class_names(yaml_path="data.yaml"):
    if not os.path.exists(yaml_path):
        st.error("data.yaml file not found.")
        st.stop()
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get("names", [])

# ✅ Load YOLOv3 ONNX model
def load_yolo_model(onnx_path="best.onnx"):
    if not os.path.exists(onnx_path):
        st.error("best.onnx not found.")
        st.stop()
    net = cv2.dnn.readNetFromONNX(onnx_path)
    return net

# ✅ Predict bounding boxes using YOLO
def predict_boxes(net, image):
    INPUT_WH_YOLO = 640
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True)
    net.setInput(blob)
    outputs = net.forward()
    return outputs[0], input_image

# ✅ Post-process YOLO output
def process_predictions(detections, input_image, class_names, conf_threshold=0.4, score_threshold=0.25):
    boxes, confidences, class_ids = [], [], []
    h, w = input_image.shape[:2]
    x_factor, y_factor = w / 640, h / 640

    for det in detections:
        conf = det[4]
        if conf > conf_threshold:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_threshold:
                cx, cy, bw, bh = det[0:4]
                left = int((cx - 0.5 * bw) * x_factor)
                top = int((cy - 0.5 * bh) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)
                boxes.append([left, top, width, height])
                confidences.append(float(conf))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes, class_ids

# ✅ Run EasyOCR on cropped regions
def run_easyocr_on_boxes(image, boxes, indices, reader):
    texts = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        x, y = max(x, 0), max(y, 0)
        crop = image[y:y+h, x:x+w]
        result = reader.readtext(crop, detail=0)
        text = result[0] if result else ""
        texts.append(text)
    return texts

# ✅ Draw boxes with labels
def draw_boxes(image, boxes, indices, class_ids, class_names):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = class_names[class_ids[i]]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
    return image

# ✅ Streamlit UI setup
st.set_page_config(page_title="Custom OCR", layout="wide")
st.title("🩺 Custom OCR for Lab Reports")
st.markdown("Upload JPG images to extract labeled medical values.")

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    class_names = load_class_names("data.yaml")
    net = load_yolo_model("best.onnx")
    reader = easyocr.Reader(["en"], gpu=False)

    for uploaded_file in uploaded_files:
        st.subheader(f"📄 File: {uploaded_file.name}")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        detections, input_image = predict_boxes(net, image)
        indices, boxes, class_ids = process_predictions(detections, input_image, class_names)

        ocr_texts = run_easyocr_on_boxes(image, boxes, indices, reader)
        labels = [class_names[class_ids[i]] for i in indices.flatten()]
        df = pd.DataFrame({"Label": labels, "Text": ocr_texts})
        st.dataframe(df)

        st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name=f"{uploaded_file.name}_ocr.csv")

        annotated = draw_boxes(image.copy(), boxes, indices, class_ids, class_names)
        st.image(annotated, caption="📌 Annotated Result", use_container_width=True)
