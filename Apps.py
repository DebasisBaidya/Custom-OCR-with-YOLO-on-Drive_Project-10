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
    model_path = 'best.onnx'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ YOLO prediction
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    input_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Process predictions
def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    output = preds[0]
    h, w = input_img.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for det in output:
        conf = det[4]
        if conf > conf_thresh:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw/2) * x_factor)
                y = int((cy - bh/2) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)
                boxes.append([x, y, width, height])
                confidences.append(float(conf))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices, boxes, class_ids

# ✅ Preprocess crop for OCR
def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshed

# ✅ Perform OCR based on class IDs
def perform_easyocr_classified(image, boxes, indices, class_ids, reader):
    data = {"Test Name": [], "Value": [], "Units": [], "Reference Range": []}

    label_map = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}

    for i in indices.flatten():
        if i >= len(boxes): continue
        x, y, w, h = boxes[i]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        crop = image[y:y2, x:x2]
        roi = preprocess_crop(crop)

        text = " ".join(reader.readtext(roi, detail=0)).strip()
        class_id = class_ids[i]
        label = label_map.get(class_id)
        if label and text:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    data[label].append(line)

    # Pad all fields to same length
    max_len = max(len(data["Test Name"]), len(data["Value"]),
                  len(data["Units"]), len(data["Reference Range"]))

    for key in data:
        data[key].extend([""] * (max_len - len(data[key])))

    df = pd.DataFrame(data)

    # ✅ Optional: highlight abnormal
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

# ✅ Draw boxes on image
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# ✅ Streamlit App
st.set_page_config(layout="wide")
st.title("🧾 Custom OCR: YOLOv5 + EasyOCR (Streamlit App)")

uploaded_files = st.file_uploader("📤 Upload Medical Report JPGs", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### 📄 Processing `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Detecting fields and extracting text..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = perform_easyocr_classified(image, boxes, indices, class_ids, reader)

        def highlight_abnormal(row):
            return ["background-color: #ffdddd" if row.get("Abnormal", False) else ""] * len(row)

        st.success("✅ Extraction Complete!")

        if "Abnormal" in df.columns:
            styled_df = df.drop(columns="Abnormal").style.apply(highlight_abnormal, axis=1)
            st.dataframe(styled_df)
        else:
            st.dataframe(df)

        st.download_button("📥 Download CSV",
                           df.drop(columns="Abnormal", errors="ignore").to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        annotated = draw_boxes(image.copy(), boxes, indices)
        st.image(annotated, caption="📦 Detected Fields", use_container_width=True)
