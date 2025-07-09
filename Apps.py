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

# ✅ Run YOLOv5 on uploaded image
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
    boxes, confidences, class_ids = [], [], []
    detections = predictions[0]
    H, W = input_image.shape[:2]
    x_factor = W / 640
    y_factor = H / 640

    for det in detections:
        confidence = det[4]
        if confidence > conf_threshold:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_threshold:
                cx, cy, w, h = det[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                boxes.append([left, top, int(w * x_factor), int(h * y_factor)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes, class_ids

# ✅ Group detected fields row-wise by y-coordinate
def group_boxes_by_row(image, boxes, indices, class_ids, reader):
    detected = []

    for i in indices.flatten():
        if i >= len(boxes) or i >= len(class_ids):
            continue
        x, y, w, h = boxes[i]
        label_id = class_ids[i]
        x, y = max(0, x), max(0, y)
        x_end, y_end = min(image.shape[1], x + w), min(image.shape[0], y + h)
        crop = image[y:y_end, x:x_end]

        result = reader.readtext(crop, detail=0)
        text = " ".join(result).strip()

        if text:
            detected.append({
                "y": y,
                "x": x,
                "label": label_id,
                "text": text
            })

    # ✅ Group detections by y position (row-wise)
    detected = sorted(detected, key=lambda d: d["y"])  # top to bottom
    grouped_rows = []
    row_threshold = 30  # pixels

    for d in detected:
        matched = False
        for row in grouped_rows:
            if abs(row["y"] - d["y"]) < row_threshold:
                row["items"].append(d)
                matched = True
                break
        if not matched:
            grouped_rows.append({"y": d["y"], "items": [d]})

    # ✅ Build structured rows
    output = []
    for row in grouped_rows:
        row_dict = {"Test Name": "", "Value": "", "Units": "", "Reference Range": ""}
        for item in row["items"]:
            if item["label"] == 0:
                row_dict["Test Name"] = item["text"]
            elif item["label"] == 1:
                row_dict["Value"] = item["text"]
            elif item["label"] == 2:
                row_dict["Units"] = item["text"]
            elif item["label"] == 3:
                row_dict["Reference Range"] = item["text"]
        output.append(row_dict)

    return pd.DataFrame(output)

# ✅ Draw detection boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

        # Row-wise grouping OCR
        df = group_boxes_by_row(image, boxes, indices, class_ids, reader)
        st.success("✅ OCR Complete!")

        st.dataframe(df.style.set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#f7f7f7"), ("color", "#333")]}
        ]).set_properties(**{"text-align": "left"}))

        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        boxed_img = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_img, caption="🔍 Detected Regions", use_container_width=True)
