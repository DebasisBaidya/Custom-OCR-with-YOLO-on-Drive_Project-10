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
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ Run YOLO
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

# ✅ Process YOLO predictions
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

# ✅ Preprocess for better OCR
def preprocess_image(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = cv2.bitwise_not(thresh)
    return roi

# ✅ Run OCR using EasyOCR + index-based mapping
def perform_ocr_easyocr(image, boxes, indices, reader):
    test_names, values, units, reference_ranges = [], [], [], []

    # ✅ Use correct index-based box mapping
    box_mapping = {
        61: "Test Name",
        14: "Value",
        26: "Units",
        41: "Reference Range"
    }

    for i in indices.flatten():
        if i >= len(boxes):
            continue
        x, y, w, h = boxes[i]
        x_end = min(x + w, image.shape[1])
        y_end = min(y + h, image.shape[0])
        crop_img = image[y:y_end, x:x_end]
        roi = preprocess_image(crop_img)

        result = reader.readtext(roi, detail=0)
        text = " ".join(result).strip()

        if i in box_mapping:
            field = box_mapping[i]
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if field == "Test Name": test_names.extend(lines)
            elif field == "Value": values.extend(lines)
            elif field == "Units": units.extend(lines)
            elif field == "Reference Range": reference_ranges.extend(lines)

    # ✅ Pad all columns to same length
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

# ✅ Draw bounding boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

        # Optional debug
        st.code(f"🔍 Box Indices Detected: {indices.flatten().tolist()}")

        df = perform_ocr_easyocr(image, boxes, indices, reader)
        st.success("✅ OCR Complete!")
        st.dataframe(df)

        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        boxed = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed, caption="🧠 Detected Regions", use_container_width=True)
