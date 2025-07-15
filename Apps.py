# --------------------------------------------------
# 🌿 I'm importing the required libraries
# --------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import re

# --------------------------------------------------
# 🧠 I'm defining class mapping for detected fields
# --------------------------------------------------
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# --------------------------------------------------
# 🧠 I'm extracting table data from image using OCR
# --------------------------------------------------
def extract_table_text(image, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    grouped_data = []

    for i in indices:
        cls = class_ids[i]
        label = class_map.get(cls, "Field")
        x, y, w, h = boxes[i]
        crop = image[y:y+h, x:x+w]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            text = reader.readtext(roi, detail=0)
        except:
            text = []

        joined = " ".join([t.strip() for t in text if t.strip()])
        if joined:
            grouped_data.append({
                "label": label,
                "x": x,
                "y_center": y + h // 2,
                "text": joined
            })

    # Sort and group rows based on Y center proximity
    grouped_data.sort(key=lambda x: x["y_center"])
    rows = []
    for item in grouped_data:
        for row in rows:
            if abs(row[0]["y_center"] - item["y_center"]) < 15:
                row.append(item)
                break
        else:
            rows.append([item])

    # Compile extracted fields row by row
    results = {k: [] for k in class_map.values()}
    for row in rows:
        row.sort(key=lambda x: x["x"])
        row_data = {k: "" for k in class_map.values()}
        for item in row:
            row_data[item["label"]] += (" " + item["text"]).strip()
        for k in class_map.values():
            results[k].append(row_data[k])

    return pd.DataFrame(results)

# --------------------------------------------------
# 🎯 I'm building the Streamlit app UI
# --------------------------------------------------
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown(
    "<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> 
    to test and upload from this: 
    <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' 
    target='_blank'>Drive Link</a></div><br>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style='text-align:center; margin-bottom:0;'>
📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b><br>
<small>📂 Please upload one or more lab report images to start extraction.</small>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=st.session_state.get("uploader_key", "file_uploader"),
)

# --------------------------------------------------
# 🧠 I'm defining YOLOv5 ONNX loading & prediction
# --------------------------------------------------
def load_yolo_model():
    model = cv2.dnn.readNetFromONNX('models/best.onnx')
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

def predict_yolo(model, image):
    input_size = 640
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (input_size, input_size), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_image

def process_predictions(preds, input_image, conf_threshold=0.4, score_threshold=0.25):
    detections = preds[0]
    boxes, confidences, class_ids = [], [], []

    image_h, image_w = input_image.shape[:2]
    x_factor = image_w / 640
    y_factor = image_h / 640

    for det in detections:
        confidence = det[4]
        if confidence > conf_threshold:
            class_scores = det[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            if class_score > score_threshold:
                cx, cy, w, h = det[:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                boxes.append([left, top, int(w * x_factor), int(h * y_factor)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    indices = indices.flatten() if len(indices) > 0 else []
    return indices, boxes, class_ids

# --------------------------------------------------
# 🎨 I'm drawing detected bounding boxes
# --------------------------------------------------
def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        cls_id = class_ids[i]
        label = class_map.get(cls_id, str(cls_id))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return image

if uploaded_files:
    model = load_yolo_model()
    for file in uploaded_files:
        st.markdown(
            f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            with st.spinner("🔍 Running YOLOv5 Detection and OCR..."):
                image = np.array(Image.open(file).convert("RGB"))
                preds, input_img = predict_yolo(model, image)
                indices, boxes, class_ids = process_predictions(preds, input_img)

                if len(indices) == 0:
                    st.warning("⚠️ No fields detected in this image.")
                    continue

                df = extract_table_text(image, boxes, indices, class_ids)

        st.markdown("<h5 style='text-align:center;'>✅ Extraction Complete!</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(image.copy(), boxes, indices, class_ids), use_container_width=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            col_dl, col_rst = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{file.name}_ocr.csv",
                    mime="text/csv",
                )
            with col_rst:
                if st.button("🧹 Clear All"):
                    st.session_state.clear()
                    st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
                    st.rerun()


            # with col_rst:
            #     if st.button("🧹 Clear All"):
            #         st.session_state["uploaded_files"] = []
            #         st.session_state["extracted_dfs"] = []
            #         st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
            #         st.rerun()
