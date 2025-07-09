import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import os

# ✅ Load YOLOv5 model
def load_model():
    model_path = "best.onnx"  # Update path if needed
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ Preprocess cropped region
def preprocess_image(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = cv2.bitwise_not(thresh)
    return roi

# ✅ YOLO Prediction
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

# ✅ Process YOLO Outputs
def process_predictions(predictions, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes = []
    confidences = []
    detections = predictions[0]
    h, w = input_image.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for i, det in enumerate(detections):
        confidence = det[4]
        if confidence > conf_threshold:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_threshold:
                cx, cy, bw, bh = det[0:4]
                x = int((cx - 0.5 * bw) * x_factor)
                y = int((cy - 0.5 * bh) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes

# ✅ Perform OCR using EasyOCR with fixed index mapping
def perform_ocr_on_crops(image, boxes, indices, reader):
    test_names, values, units, reference_ranges = [], [], [], []

    box_mapping = {
        8: "Test Name",
        18: "Value",
        13: "Units",
        37: "Reference Range"
    }

    for i in indices.flatten():
        if i not in box_mapping:
            continue

        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)
        x_end = min(x + w, image.shape[1])
        y_end = min(y + h, image.shape[0])
        crop_img = image[y:y_end, x:x_end]
        roi = preprocess_image(crop_img)
        text = " ".join(reader.readtext(roi, detail=0)).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        field_name = box_mapping[i]
        for line in lines:
            if field_name == "Test Name": test_names.append(line)
            elif field_name == "Value": values.append(line)
            elif field_name == "Units": units.append(line)
            elif field_name == "Reference Range": reference_ranges.append(line)

    # ✅ Equal length padding
    max_len = max(len(test_names), len(values), len(units), len(reference_ranges))
    test_names += [""] * (max_len - len(test_names))
    values += [""] * (max_len - len(values))
    units += [""] * (max_len - len(units))
    reference_ranges += [""] * (max_len - len(reference_ranges))

    return pd.DataFrame({
        "Test Name": test_names,
        "Value": values,
        "Units": units,
        "Reference Range": reference_ranges
    })

# ✅ Draw boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# ✅ Streamlit App UI
st.set_page_config(layout="wide")
st.title("🧠 Medical Report OCR – YOLOv5 + EasyOCR")

uploaded_files = st.file_uploader("📤 Upload JPG/PNG images", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for uploaded_file in uploaded_files:
        st.markdown(f"### 📄 File: `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        preds, input_img = predict_yolo(model, image)
        indices, boxes = process_predictions(preds, input_img)

        st.code(f"🔍 Box Indices Detected: {indices.flatten().tolist()}")

        if len(indices) == 0:
            st.warning("⚠️ No detections found.")
            continue

        df = perform_ocr_on_crops(image, boxes, indices, reader)
        st.success("✅ OCR Done!")
        st.dataframe(df)

        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        boxed = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed, caption="🖼️ Detected Boxes", use_container_width=True)
