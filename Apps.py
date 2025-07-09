import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import os
import re

# ✅ Load YOLOv5 ONNX model
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
    H, W = image.shape[:2]
    max_rc = max(H, W)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:H, 0:W] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_image

# ✅ Safe box filtering
def process_predictions(preds, image, conf_threshold=0.4, score_threshold=0.25):
    boxes = []
    confidences = []
    detections = preds[0]
    H, W = image.shape[:2]
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
                w = int(w * x_factor)
                h = int(h * y_factor)
                boxes.append([x, y, w, h])
                confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    if len(indices) == 0:
        return []
    flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
    return [boxes[i] for i in flat_indices]

# ✅ Preprocess image for OCR
def preprocess_crop(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshed

# ✅ OCR + vertical clustering
def extract_table_by_alignment(image, boxes, reader):
    fields = []

    for box in boxes:
        x, y, w, h = box
        crop = image[y:y+h, x:x+w]
        roi = preprocess_crop(crop)
        text = " ".join(reader.readtext(roi, detail=0)).strip()
        if text:
            fields.append({
                "text": text,
                "x": x,
                "y": y,
                "cy": y + h // 2
            })

    # Group vertically
    fields.sort(key=lambda d: d["cy"])
    row_threshold = 25
    rows = []

    for field in fields:
        placed = False
        for row in rows:
            if abs(row["cy"] - field["cy"]) < row_threshold:
                row["fields"].append(field)
                row["cy_vals"].append(field["cy"])
                row["cy"] = int(np.mean(row["cy_vals"]))
                placed = True
                break
        if not placed:
            rows.append({
                "cy": field["cy"],
                "cy_vals": [field["cy"]],
                "fields": [field]
            })

    # Sort row items by x
    final_data = []
    for row in rows:
        sorted_row = sorted(row["fields"], key=lambda d: d["x"])
        texts = [f["text"] for f in sorted_row]
        while len(texts) < 4:
            texts.append("")
        final_data.append({
            "Test Name": texts[0],
            "Value": texts[1],
            "Units": texts[2],
            "Reference Range": texts[3]
        })

    df = pd.DataFrame(final_data)

    def is_abnormal(row):
        try:
            val = float(row["Value"].replace(",", "."))
            rng = re.findall(r"[\\d.]+", row["Reference Range"])
            if len(rng) >= 2:
                low, high = float(rng[0]), float(rng[1])
                return not (low <= val <= high)
        except:
            return False
        return False

    df["Abnormal"] = df.apply(is_abnormal, axis=1)
    return df

# ✅ Draw Boxes
def draw_boxes(image, boxes):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# ✅ Streamlit App
st.set_page_config(layout="wide")
st.title("📊 OCR for Lab Reports (YOLO + EasyOCR + Box Clustering)")

uploaded_files = st.file_uploader("📤 Upload JPG/PNG image(s)", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for uploaded_file in uploaded_files:
        st.markdown(f"### 📄 File: `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        with st.spinner("🔍 Running OCR..."):
            preds, input_img = predict_yolo(model, image)
            boxes = process_predictions(preds, input_img)

            if not boxes:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_table_by_alignment(image, boxes, reader)

        def highlight_abnormal(row):
            return ["background-color: #ffdddd" if row["Abnormal"] else ""] * len(row)

        st.success("✅ Extraction Complete!")
        st.dataframe(df.drop(columns="Abnormal").style.apply(highlight_abnormal, axis=1))

        st.download_button("📥 Download CSV", df.drop(columns="Abnormal").to_csv(index=False),
                           file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        boxed_img = draw_boxes(image.copy(), boxes)
        st.image(boxed_img, caption="📦 Detected Fields", use_container_width=True)
