# ✅ Streamlit App for Project 10 - Custom OCR (YOLOv5 + EasyOCR)
# Purpose: Detect test fields from medical reports and extract structured data with OCR.

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image
import easyocr

# ✅ Class names aligned with data.yaml
CLASS_NAMES = ["Test Name", "Value", "Units", "Reference Range"]

# ✅ Load YOLOv5 ONNX model
@st.cache_resource
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please check.")
        st.stop()
    model = cv2.dnn.readNet(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ Perform inference
def run_inference(model, image):
    INPUT_SIZE = 640
    h, w = image.shape[:2]
    scale = max(h, w)
    padded = np.zeros((scale, scale, 3), dtype=np.uint8)
    padded[:h, :w] = image
    blob = cv2.dnn.blobFromImage(padded, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, padded

# ✅ Extract boxes and labels
def extract_boxes(preds, image, conf_thresh=0.4, score_thresh=0.25):
    boxes, class_ids, confidences = [], [], []
    h, w = image.shape[:2]
    x_factor, y_factor = w / 640, h / 640
    detections = preds[0]
    for det in detections:
        conf = det[4]
        if conf > conf_thresh:
            scores = det[5:]
            class_id = np.argmax(scores)
            score = scores[class_id]
            if score > score_thresh:
                cx, cy, bw, bh = det[0:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                w_box = int(bw * x_factor)
                h_box = int(bh * y_factor)
                boxes.append([x, y, w_box, h_box])
                class_ids.append(class_id)
                confidences.append(float(conf))
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    results = []
    for i in idxs.flatten():
        results.append({
            "box": boxes[i],
            "class_id": class_ids[i],
            "label": CLASS_NAMES[class_ids[i]]
        })
    return results

# ✅ EasyOCR Reader
@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(['en'])

# ✅ Apply OCR to each crop
def run_easyocr(image, results, reader):
    output = {label: [] for label in CLASS_NAMES}
    for res in results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        crop = image[y:y+h, x:x+w]
        text = reader.readtext(crop, detail=0)
        joined = " ".join(text).strip()
        output[res['label']].append(joined)
    max_rows = max(len(v) for v in output.values())
    for key in output:
        while len(output[key]) < max_rows:
            output[key].append("")
    return pd.DataFrame(output)

# ✅ Draw bounding boxes on image
def draw_boxes(image, results):
    for res in results:
        x, y, w, h = res['box']
        label = res['label']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return image

# ✅ Streamlit UI
st.set_page_config(layout="wide")
st.title("🧾 Custom OCR for Medical Reports (YOLOv5 + EasyOCR)")

uploaded_files = st.file_uploader("Upload Medical Report Image(s) (JPG/PNG)", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = get_easyocr_reader()
    for uploaded_file in uploaded_files:
        st.markdown(f"### Processing: `{uploaded_file.name}`")
        with st.spinner("🔍 Detecting and extracting data..."):
            image = np.array(Image.open(uploaded_file).convert("RGB"))
            preds, padded = run_inference(model, image)
            results = extract_boxes(preds, padded)
            ocr_df = run_easyocr(image, results, reader)

        st.subheader("📋 Detected Regions")
        st.dataframe(ocr_df, use_container_width=True)

        st.download_button(
            label="📥 Download CSV",
            data=ocr_df.to_csv(index=False),
            file_name=f"{uploaded_file.name}_ocr.csv",
            mime='text/csv'
        )

        final_img = draw_boxes(image.copy(), results)
        st.image(final_img, caption=f"📌 Annotated Image for {uploaded_file.name}", use_container_width=True)
