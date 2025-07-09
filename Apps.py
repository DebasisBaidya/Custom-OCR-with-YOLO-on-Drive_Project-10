import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import easyocr

# ✅ Use your original YOLO box index → field mapping
box_mapping = {
    36: "Test Name",
    27: "Value",
    29: "Units",
    34: "Reference Range"
}

# ✅ Load YOLOv5 ONNX model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ Run YOLO detection
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    input_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Process YOLO outputs
def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes = []
    confidences = []
    detections = preds[0]
    h, w = input_img.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for i, det in enumerate(detections):
        conf = det[4]
        if conf > conf_thresh:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes

# ✅ Perform OCR using EasyOCR with explode logic
def extract_fields(image, boxes, indices, reader):
    data = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    for i in indices:
        if i >= len(boxes):
            continue
        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(x + w, image.shape[1]), min(y + h, image.shape[0])
        crop = image[y:y2, x:x2]

        # Preprocessing like main.py
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            ocr_lines = reader.readtext(roi, detail=0)
        except:
            ocr_lines = []

        cleaned = [line.strip() for line in ocr_lines if line.strip()]
        if i in box_mapping:
            label = box_mapping[i]
            data[label].extend(cleaned)

    # Explode to rows
    df = pd.DataFrame({col: pd.Series(values) for col, values in data.items()})
    return df

# ✅ Draw bounding boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Streamlit App UI
st.set_page_config(layout="wide")
st.title("🧪 Medical Report OCR (YOLOv5 + EasyOCR + main.py logic)")

uploaded_files = st.file_uploader("📤 Upload JPG files", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for uploaded_file in uploaded_files:
        st.markdown(f"### 📄 `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        with st.spinner("🔍 Detecting and Extracting..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_fields(image, boxes, indices, reader)

        st.success("✅ OCR Complete!")
        st.dataframe(df)

        st.download_button("📥 Download CSV",
                           df.to_csv(index=False),
                           file_name=f"{uploaded_file.name}_ocr.csv",
                           mime="text/csv")

        boxed_image = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_image, caption="📦 Detected Fields", use_container_width=True)
