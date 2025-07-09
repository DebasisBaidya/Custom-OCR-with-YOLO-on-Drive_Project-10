import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import os

# ✅ Your original mapping logic
box_mapping = {
    36: "Test Name",
    27: "Value",
    29: "Units",
    34: "Reference Range"
}

# ✅ Load YOLOv5 model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ Run YOLO prediction
def predict_yolo(model, image):
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Post-process boxes
def process_predictions(preds, input_img, conf_threshold=0.4, score_threshold=0.25):
    boxes, confidences = [], []
    detections = preds[0]
    h, w = input_img.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for i, det in enumerate(detections):
        conf = det[4]
        if conf > conf_threshold:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_threshold:
                cx, cy, bw, bh = det[0:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes

# ✅ Your original EasyOCR logic (explode fields)
def extract_fields(image, boxes, indices, reader):
    results = {
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
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        crop = image[y:y2, x:x2]

        # Preprocess (as in main.py)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(thresh)

        try:
            ocr_lines = reader.readtext(roi, detail=0)
        except:
            ocr_lines = []

        text_lines = [line.strip() for line in ocr_lines if line.strip()]
        if i in box_mapping:
            label = box_mapping[i]
            results[label].extend(text_lines)

    # Build DataFrame by exploding each column
    df = pd.DataFrame({col: pd.Series(vals) for col, vals in results.items()})
    return df

# ✅ Draw bounding boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# ✅ Streamlit App
st.set_page_config(layout="wide")
st.title("📋 Medical Report OCR (Original Logic from main.py)")

uploaded_files = st.file_uploader("📤 Upload JPG image(s)", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### Processing: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Running Detection + OCR..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_fields(image, boxes, indices, reader)

        st.success("✅ Extraction Complete!")
        st.dataframe(df)

        st.download_button("📥 Download CSV",
                           df.to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        boxed_img = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_img, caption="📦 Detected Fields", use_container_width=True)
