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
        st.error(f"❌ Model not found at {model_path}")
        st.stop()
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ Run YOLOv5
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

# ✅ Extract boxes, scores, class_ids
def process_predictions(preds, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes, scores, class_ids = [], [], []
    detections = preds[0]
    h, w = input_image.shape[:2]
    x_factor, y_factor = w / 640, h / 640

    for det in detections:
        confidence = det[4]
        if confidence > conf_threshold:
            class_scores = det[5:]
            class_id = np.argmax(class_scores)
            if class_scores[class_id] > score_threshold:
                cx, cy, bw, bh = det[:4]
                x = int((cx - 0.5 * bw) * x_factor)
                y = int((cy - 0.5 * bh) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                scores.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, 0.45)
    return indices, boxes, class_ids

# ✅ Perform OCR and assign per class
def perform_ocr(image, boxes, indices, class_ids, reader):
    field_texts = {0: [], 1: [], 2: [], 3: []}  # 0-Test Name, 1-Value, 2-Unit, 3-Ref

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cls_id = class_ids[i]
        x, y = max(0, x), max(0, y)
        crop = image[y:y+h, x:x+w]
        result = reader.readtext(crop, detail=0)
        text = " ".join(result).strip()
        field_texts[cls_id].append((y, text))  # store with y for sorting

    # Sort all by y-position to keep top-to-bottom order
    for k in field_texts:
        field_texts[k] = [txt for y, txt in sorted(field_texts[k], key=lambda x: x[0])]

    # Pad all fields to same length
    max_len = max(len(v) for v in field_texts.values())
    for k in field_texts:
        field_texts[k] += [""] * (max_len - len(field_texts[k]))

    return pd.DataFrame({
        "Test Name": field_texts[0],
        "Value": field_texts[1],
        "Units": field_texts[2],
        "Reference Range": field_texts[3]
    })

# ✅ Draw boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# ✅ Streamlit UI
st.set_page_config(layout="wide")
st.title("🩺 Custom OCR for Medical Reports (YOLOv5 + EasyOCR)")

uploaded_files = st.file_uploader("📤 Upload JPG/PNG image(s)", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for uploaded_file in uploaded_files:
        st.markdown(f"### 📌 Processing: `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        preds, input_img = predict_yolo(model, image)
        indices, boxes, class_ids = process_predictions(preds, input_img)

        if len(indices) == 0:
            st.warning("⚠️ No text regions detected!")
            continue

        df = perform_ocr(image, boxes, indices, class_ids, reader)
        st.success("✅ OCR Complete!")

        st.dataframe(df.style.set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#f7f7f7"), ("color", "#333")]}
        ]).set_properties(**{"text-align": "left"}))

        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        boxed = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed, caption="🔍 Detected Regions", use_container_width=True)
