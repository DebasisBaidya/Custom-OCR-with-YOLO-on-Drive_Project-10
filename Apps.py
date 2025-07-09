import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import easyocr
import re

# ✅ YOLO class labels
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
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

# ✅ Run YOLO prediction
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    input_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Extract boxes, scores, classes
def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    h, w = input_img.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for det in preds[0]:
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

# ✅ Extract and group rows by Y position
def extract_rows(image, boxes, indices, class_ids, reader):
    detections = []

    for i in indices:
        x, y, w, h = boxes[i]
        crop = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            text = " ".join(reader.readtext(roi, detail=0)).strip()
        except:
            text = ""

        if text:
            detections.append({
                "label": class_map.get(class_ids[i], "Unknown"),
                "text": text,
                "cy": y + h // 2
            })

    # ✅ Group into rows by Y position
    detections.sort(key=lambda d: d["cy"])
    row_threshold = 35
    rows = []

    for det in detections:
        placed = False
        for row in rows:
            if abs(row["cy"] - det["cy"]) < row_threshold:
                row["items"].append(det)
                row["cy_vals"].append(det["cy"])
                row["cy"] = int(np.mean(row["cy_vals"]))
                placed = True
                break
        if not placed:
            rows.append({"cy": det["cy"], "cy_vals": [det["cy"]], "items": [det]})

    # ✅ Build clean DataFrame rows
    final_data = []
    for row in rows:
        test_name_parts = [i["text"] for i in row["items"] if i["label"] == "Test Name"]
        value = next((i["text"] for i in row["items"] if i["label"] == "Value"), "")
        units = next((i["text"] for i in row["items"] if i["label"] == "Units"), "")
        ref_range = next((i["text"] for i in row["items"] if i["label"] == "Reference Range"), "")

        final_data.append({
            "Test Name": " ".join(test_name_parts).strip(),
            "Value": value,
            "Units": units,
            "Reference Range": ref_range
        })

    df = pd.DataFrame(final_data)

    # ✅ Optional: Flag abnormal values
    def is_abnormal(row):
        try:
            val = float(row["Value"].replace(",", "."))
            matches = re.findall(r"[\d.]+", row["Reference Range"])
            if len(matches) >= 2:
                low, high = float(matches[0]), float(matches[1])
                return not (low <= val <= high)
        except:
            return False
        return False

    df["Abnormal"] = df.apply(is_abnormal, axis=1)
    return df

# ✅ Draw YOLO boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Streamlit App
st.set_page_config(layout="wide")
st.title("🧪 Lab Report OCR – ✅ Final One-Line Fix")

uploaded_files = st.file_uploader("📤 Upload JPG image(s)", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### 📄 File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Extracting..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_rows(image, boxes, indices, class_ids, reader)

        st.success("✅ Extraction Complete!")
        st.dataframe(df.drop(columns="Abnormal"))

        st.download_button("📥 Download CSV",
                           df.drop(columns="Abnormal").to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        boxed_img = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_img, caption="📦 Detected Fields", use_container_width=True)
