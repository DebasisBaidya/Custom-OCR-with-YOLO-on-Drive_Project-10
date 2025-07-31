# --------------------------------------------------
# 🏗️ I'm importing the required libraries
# --------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
from collections import defaultdict

# --------------------------------------------------
# 🧠 Class mapping for YOLO model's output
# --------------------------------------------------
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# --------------------------------------------------
# 📦 I'm loading the YOLOv3 ONNX model
# --------------------------------------------------
net = cv2.dnn.readNetFromONNX("best.onnx")

# --------------------------------------------------
# 🧠 I'm initializing EasyOCR reader
# --------------------------------------------------
reader = easyocr.Reader(['en'])

# --------------------------------------------------
# 🔍 I'm defining helper to run YOLO detection
# --------------------------------------------------
def detect_fields(image):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0]

    boxes, confidences, class_ids = [], [], []
    rows = outputs.shape[0]
    img_h, img_w = image.shape[:2]

    for i in range(rows):
        row = outputs[i]
        confidence = row[4]
        if confidence >= 0.4:
            class_scores = row[5:]
            class_id = np.argmax(class_scores)
            if class_scores[class_id] > 0.4:
                cx, cy, w, h = row[0:4]
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
    final = []
    for i in indices.flatten():
        x1, y1, x2, y2 = boxes[i]
        final.append((class_ids[i], confidences[i], (x1, y1, x2, y2)))
    return final

# --------------------------------------------------
# 🧠 I'm defining helper to run EasyOCR on crops
# --------------------------------------------------
def run_ocr_on_boxes(image, boxes):
    field_data = defaultdict(list)
    for class_id, conf, (x1, y1, x2, y2) in boxes:
        crop = image[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        if crop.size == 0: continue
        text = reader.readtext(crop, detail=0, paragraph=True)
        clean_text = ' '.join([t.strip() for t in text if t.strip()])
        if clean_text:
            field_data[class_map[class_id]].append((y1, clean_text))  # y1 for sorting vertically
    return field_data

# --------------------------------------------------
# 📊 I'm formatting and sorting results
# --------------------------------------------------
def format_results(field_data):
    rows = []
    combined = defaultdict(dict)
    for label, values in field_data.items():
        for y, text in values:
            combined[y][label] = text
    for y in sorted(combined.keys()):
        row = combined[y]
        rows.append([
            row.get("Test Name", ""),
            row.get("Value", ""),
            row.get("Units", ""),
            row.get("Reference Range", "")
        ])
    df = pd.DataFrame(rows, columns=["Test Name", "Value", "Units", "Reference Range"])
    return df

# --------------------------------------------------
# 🎯 Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="🧾 Lab Report Extractor", layout="centered")
st.title("🧠 Lab Report to Table")
st.markdown("Upload a **Thyrocare** JPG image to auto-extract test data.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(image, caption="Uploaded Report", use_column_width=True)

    with st.spinner("🔍 Detecting fields..."):
        detections = detect_fields(image)
        results = run_ocr_on_boxes(image, detections)
        df = format_results(results)

    if not df.empty:
        st.success("✅ Extraction Complete")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ No data extracted. Try a clearer image.")
