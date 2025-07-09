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

# ✅ Process predictions and return class_ids
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
                w_box = int(bw * x_factor)
                h_box = int(bh * y_factor)
                boxes.append([x, y, w_box, h_box])
                confidences.append(float(conf))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices, boxes, class_ids

# ✅ Preprocess crop
def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshed

# ✅ Perform OCR and store by class label (explode-based)
def extract_by_class_explode(image, boxes, indices, class_ids, reader):
    output = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    class_map = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}

    for idx in indices.flatten():
        if idx >= len(boxes):
            continue

        x, y, w, h = boxes[idx]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(x + w, image.shape[1]), min(y + h, image.shape[0])
        crop = image[y:y2, x:x2]
        roi = preprocess_crop(crop)

        text_lines = reader.readtext(roi, detail=0)
        clean_text = [line.strip() for line in text_lines if line.strip()]
        class_id = class_ids[idx]
        label = class_map.get(class_id)

        if label and clean_text:
            output[label].extend(clean_text)

    # Convert to exploded rows
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in output.items()]))

    # Optional: detect abnormal values
    def is_abnormal(row):
        try:
            val = float(row["Value"].replace(",", "."))
            rng = re.findall(r"[\d.]+", str(row["Reference Range"]))
            if len(rng) >= 2:
                low, high = float(rng[0]), float(rng[1])
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
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# ✅ Streamlit UI
st.set_page_config(layout="wide")
st.title("🧾 Streamlit OCR App (YOLOv5 + EasyOCR + Explode Logic)")

uploaded_files = st.file_uploader("📤 Upload JPG medical report(s)", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### 📄 File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Running detection and OCR..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_by_class_explode(image, boxes, indices, class_ids, reader)

        # Highlight abnormal values
        def highlight_abnormal(row):
            return ["background-color: #ffdddd" if row.get("Abnormal") else ""] * len(row)

        st.success("✅ Extraction Complete!")

        if "Abnormal" in df.columns:
            styled_df = df.drop(columns="Abnormal").style.apply(highlight_abnormal, axis=1)
            st.dataframe(styled_df)
        else:
            st.dataframe(df)

        # CSV export
        st.download_button("📥 Download CSV",
                           df.drop(columns="Abnormal", errors="ignore").to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        # Show image with bounding boxes
        boxed_img = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_img, caption="📦 Detected Fields", use_container_width=True)
