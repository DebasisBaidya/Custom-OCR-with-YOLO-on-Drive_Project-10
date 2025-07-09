import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import os

# ✅ Load YOLOv5 model
def load_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ Predict using YOLO
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

# ✅ Process YOLO outputs
def process_predictions(predictions, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes, confidences = [], []
    detections = predictions[0]
    H, W = input_image.shape[:2]
    x_factor = W / 640
    y_factor = H / 640

    for i, det in enumerate(detections):
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

# ✅ Run EasyOCR and map using box index
def perform_ocr_easyocr(image, boxes, indices, reader):
    data = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    # Index-based box mapping like app (12).py
    box_mapping = {
        36: "Test Name",
        27: "Value",
        29: "Units",
        34: "Reference Range"
    }

    for i in indices.flatten():
        if i >= len(boxes):
            continue
        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)
        x_end = min(x + w, image.shape[1])
        y_end = min(y + h, image.shape[0])
        crop = image[y:y_end, x:x_end]

        result = reader.readtext(crop, detail=0)
        text = " ".join(result).strip()

        if i in box_mapping:
            field = box_mapping[i]
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            data[field].extend(lines)

    # Pad fields to equal length
    max_len = max(len(v) for v in data.values())
    for k in data:
        data[k].extend([""] * (max_len - len(data[k])))

    return pd.DataFrame(data)

# ✅ Draw detected boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# ✅ Streamlit UI
st.set_page_config(layout="wide")
st.title("🩺 Custom OCR for Lab Reports (YOLOv5 + EasyOCR + Index Mapping)")

uploaded_files = st.file_uploader("📤 Upload JPG/PNG image(s)", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for uploaded_file in uploaded_files:
        st.markdown(f"### 📌 Processing: `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        preds, input_img = predict_yolo(model, image)
        indices, boxes = process_predictions(preds, input_img)

        if len(indices) == 0:
            st.warning("⚠️ No regions detected.")
            continue

        df = perform_ocr_easyocr(image, boxes, indices, reader)
        st.success("✅ OCR Complete!")

        st.dataframe(df)

        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        boxed = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed, caption="🔍 Detected Regions", use_container_width=True)
