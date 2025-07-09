import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import os
import re
from itertools import zip_longest

# ✅ Load YOLOv5 model
def load_yolo_model():
    model_path = 'best.onnx'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    model = cv2.dnn.readNet(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ Predict using YOLO
def predict_yolo(model, image):
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_image

# ✅ Process YOLO detections
def process_predictions(preds, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes, confidences, class_ids = [], [], []
    detections = preds[0]
    H, W = input_image.shape[:2]
    x_factor = W / 640
    y_factor = H / 640

    for det in detections:
        conf = det[4]
        if conf > conf_threshold:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_threshold:
                cx, cy, w, h = det[:4]
                x = int((cx - 0.5 * w) * x_factor)
                y = int((cy - 0.5 * h) * y_factor)
                boxes.append([x, y, int(w * x_factor), int(h * y_factor)])
                confidences.append(float(conf))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes, class_ids

# ✅ Preprocess for OCR
def preprocess_image(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = cv2.bitwise_not(thresh)
    return roi

# ✅ Extract fields with smart row alignment
def extract_fields_smart(image, boxes, indices, class_ids, reader):
    field_labels = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}
    detections = []

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        crop = image[y:y+h, x:x+w]
        roi = preprocess_image(crop)
        text = " ".join(reader.readtext(roi, detail=0)).strip()
        if text:
            detections.append({
                "label": field_labels.get(class_ids[i], f"Class {class_ids[i]}"),
                "text": text,
                "y": y,
                "cy": y + h // 2
            })

    detections.sort(key=lambda d: d["cy"])
    row_thresh = 35
    rows = []

    for det in detections:
        placed = False
        for row in rows:
            if abs(row["cy"] - det["cy"]) < row_thresh:
                row["fields"].append(det)
                row["cy_vals"].append(det["cy"])
                row["cy"] = int(np.mean(row["cy_vals"]))
                placed = True
                break
        if not placed:
            rows.append({"cy": det["cy"], "cy_vals": [det["cy"]], "fields": [det]})

    final_rows = []
    for row in rows:
        row_data = {"Test Name": "", "Value": "", "Units": "", "Reference Range": ""}
        for field in row["fields"]:
            label = field["label"]
            if label in row_data and row_data[label] == "":
                row_data[label] = field["text"]
        final_rows.append(row_data)

    df = pd.DataFrame(final_rows)

    def is_abnormal(row):
        try:
            val = float(row["Value"].replace(",", "."))
            range_match = re.findall(r"[\d.]+", row["Reference Range"])
            if len(range_match) >= 2:
                low, high = float(range_match[0]), float(range_match[1])
                return not (low <= val <= high)
        except:
            return False
        return False

    df["Abnormal"] = df.apply(is_abnormal, axis=1)
    return df

# ✅ Draw boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Streamlit UI
st.set_page_config(layout="wide")
st.title("🧠 Custom Medical OCR App (YOLOv5 + EasyOCR + Row Logic)")

uploaded_files = st.file_uploader("📤 Upload JPG or PNG", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for uploaded_file in uploaded_files:
        st.markdown(f"### 🔍 Processing: `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        preds, input_image = predict_yolo(model, image)
        indices, boxes, class_ids = process_predictions(preds, input_image)

        if len(indices) == 0:
            st.warning("⚠️ No detections found.")
            continue

        df = extract_fields_smart(image, boxes, indices, class_ids, reader)
        st.success("✅ Extraction Complete")

        def highlight_abnormal(row):
            return ["background-color: #ffdddd" if row.get("Abnormal") else ""] * len(row)

        st.dataframe(df.drop(columns="Abnormal").style.apply(highlight_abnormal, axis=1))
        st.download_button("📥 Download CSV", df.drop(columns="Abnormal").to_csv(index=False),
                           file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        result_img = draw_boxes(image.copy(), boxes, indices)
        st.image(result_img, caption="📦 Detected Fields", use_container_width=True)
