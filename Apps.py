# ✅ Streamlit App for Project 10 - Custom OCR with EasyOCR
# Purpose: Detect medical test labels using YOLOv3 (ONNX) and extract structured text using EasyOCR

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image
import easyocr

# ✅ Load class labels (assumes trained YOLO model matches these labels)
CLASS_NAMES = ["Test Name", "Value", "Units", "Reference Range"]

# ✅ Load YOLOv3 model from ONNX format
@st.cache_resource
def load_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please upload or check the file path.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ Run YOLO prediction
@st.cache_resource
def predict_yolo(_model, image):
    INPUT_WH_YOLO = 640
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
    _model.setInput(blob)
    preds = _model.forward()
    return preds[0], input_image

# ✅ Filter predictions and apply NMS
def process_predictions(preds, image_shape, conf_threshold=0.4, nms_threshold=0.45):
    boxes, class_ids, confidences = [], [], []
    h, w = image_shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for detection in preds:
        conf = detection[4]
        if conf > conf_threshold:
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            if class_score > conf_threshold:
                cx, cy, bw, bh = detection[0:4]
                left = int((cx - 0.5 * bw) * x_factor)
                top = int((cy - 0.5 * bh) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)
                boxes.append([left, top, width, height])
                class_ids.append(class_id)
                confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return indices, boxes, class_ids

# ✅ Extract text using EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

def ocr_with_easyocr(image):
    result = reader.readtext(image)
    text = " ".join([res[1] for res in result])
    return text.strip()

# ✅ Build Streamlit UI
st.set_page_config(layout="wide")
st.title("🩺 Custom OCR for Medical Lab Reports")

uploaded_files = st.file_uploader("Upload medical lab report JPG(s)", type="jpg", accept_multiple_files=True)

if uploaded_files:
    model = load_model()

    for uploaded_file in uploaded_files:
        st.markdown(f"### 📄 Processing: {uploaded_file.name}")
        with st.spinner("🔄 Detecting and extracting data..."):
            image = np.array(Image.open(uploaded_file))
            preds, input_image = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, image.shape)

            data_dict = {label: [] for label in CLASS_NAMES}

            for idx in indices.flatten():
                x, y, w, h = boxes[idx]
                class_id = class_ids[idx]
                label = CLASS_NAMES[class_id]
                crop = image[y:y+h, x:x+w]
                text = ocr_with_easyocr(crop)
                data_dict[label].append(text)

            # ✅ Equalize lengths for proper DataFrame formatting
            max_len = max(len(v) for v in data_dict.values())
            for k in data_dict:
                while len(data_dict[k]) < max_len:
                    data_dict[k].append("")

            df = pd.DataFrame(data_dict)

            st.success("✅ Extraction complete!")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="📥 Download CSV",
                data=df.to_csv(index=False),
                file_name=f"{uploaded_file.name}_ocr.csv",
                mime="text/csv"
            )
