import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import easyocr
import re

# ✅ Load YOLOv5 ONNX model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ Predict using YOLOv5
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    input_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Process YOLO predictions
def process_predictions(preds, input_image, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    detections = preds[0]
    h, w = input_image.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for det in detections:
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
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes, class_ids

# ✅ Preprocess crop for OCR
def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ✅ Extract fields and group them row-wise
def extract_fields_by_row(image, boxes, indices, class_ids, reader):
    class_map = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}
    fields = []

    for i in indices:
        x, y, w, h = boxes[i]
        crop = image[y:y+h, x:x+w]
        roi = preprocess_crop(crop)
        text = " ".join(reader.readtext(roi, detail=0)).strip()
        cy = y + h // 2
        if text:
            fields.append({
                "label": class_map.get(class_ids[i], f"Class {class_ids[i]}"),
                "text": text,
                "x": x,
                "cy": cy
            })

    # ✅ Group into rows by Y-position (cy)
    fields.sort(key=lambda f: f["cy"])
    rows = []
    row_threshold = 30  # pixel gap allowed vertically

    for field in fields:
        placed = False
        for row in rows:
            if abs(row["cy"] - field["cy"]) <= row_threshold:
                row["fields"].append(field)
                row["cy_vals"].append(field["cy"])
                row["cy"] = int(np.mean(row["cy_vals"]))
                placed = True
                break
        if not placed:
            rows.append({"cy": field["cy"], "cy_vals": [field["cy"]], "fields": [field]})

    # ✅ Convert grouped rows to structured DataFrame
    final_data = []
    for row in rows:
        this_row = {"Test Name": "", "Value": "", "Units": "", "Reference Range": ""}
        for f in sorted(row["fields"], key=lambda x: x["x"]):
            label = f["label"]
            if label in this_row and this_row[label] == "":
                this_row[label] = f["text"]
        final_data.append(this_row)

    df = pd.DataFrame(final_data)

    # ✅ Optional: mark abnormal values
    def is_abnormal(row):
        try:
            val = float(row["Value"].replace(",", "."))
            rng = re.findall(r"[\d.]+", row["Reference Range"])
            if len(rng) >= 2:
                low, high = float(rng[0]), float(rng[1])
                return not (low <= val <= high)
        except:
            return False
        return False

    df["Abnormal"] = df.apply(is_abnormal, axis=1)
    return df

# ✅ Draw bounding boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# ✅ Streamlit app UI
st.set_page_config(layout="wide")
st.title("📄 Medical Lab Report OCR (YOLOv5 + EasyOCR + Smart Row Matching)")

uploaded_files = st.file_uploader("📤 Upload JPG image(s)", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(["en"], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### 🖼️ File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Detecting fields and extracting text..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_fields_by_row(image, boxes, indices, class_ids, reader)

        def highlight_abnormal(row):
            return ["background-color: #ffdddd" if row.get("Abnormal") else ""] * len(row)

        st.success("✅ Extraction Complete!")

        if "Abnormal" in df.columns:
            st.dataframe(df.drop(columns="Abnormal").style.apply(highlight_abnormal, axis=1))
        else:
            st.dataframe(df)

        st.download_button("📥 Download CSV",
                           df.drop(columns="Abnormal", errors="ignore").to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        annotated = draw_boxes(image.copy(), boxes, indices)
        st.image(annotated, caption="📦 Detected Fields", use_container_width=True)
