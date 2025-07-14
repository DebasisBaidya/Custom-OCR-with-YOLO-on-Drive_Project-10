import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr  # EasyOCR import

# Field mapping
field_map = {0: 'Test Name', 1: 'Value', 2: 'Units', 3: 'Reference Range'}

def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

def predict_yolo(model, image):
    wh = 640
    h, w, _ = image.shape
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square[:h, :w] = image
    blob = cv2.dnn.blobFromImage(square, 1/255, (wh, wh), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, square

def process_predictions(preds, image, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    h, w = image.shape[:2]
    x_factor, y_factor = w / 640, h / 640
    for det in preds[0]:
        conf = det[4]
        if conf > conf_thresh:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            score = scores[class_id]
            if score > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    if indices is None or len(indices) == 0:
        return [], boxes, class_ids
    return indices.flatten(), boxes, class_ids

# Group boxes by vertical center to form rows
def group_boxes_by_rows(boxes, class_ids, threshold=15):
    rows = []
    for i, box in enumerate(boxes):
        x, y, w, h = box
        center_y = y + h/2
        placed = False
        for row in rows:
            if abs(row['center_y'] - center_y) < threshold:
                row['boxes'].append((i, box, class_ids[i]))
                row['center_y'] = np.mean([b[1][1] + b[1][3]/2 for b in row['boxes']])
                placed = True
                break
        if not placed:
            rows.append({'center_y': center_y, 'boxes': [(i, box, class_ids[i])]})
    rows.sort(key=lambda r: r['center_y'])
    return rows

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA

# OCR text extraction from a single box using EasyOCR
def ocr_text_from_box(image, box):
    x, y, w, h = box
    crop = image[y:y+h, x:x+w]
    # Convert BGR to RGB for EasyOCR
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # Run EasyOCR on cropped image
    results = reader.readtext(crop_rgb, detail=0, paragraph=True)
    text = " ".join(results).strip()
    return text

# Extract data rows aligned by detected rows and fields
def extract_rows(image, boxes, class_ids):
    rows = group_boxes_by_rows(boxes, class_ids)
    data_rows = []
    for row in rows:
        sorted_boxes = sorted(row['boxes'], key=lambda b: b[1][0])  # sort left to right
        row_data = {v: "" for v in ['Test Name', 'Value', 'Units', 'Reference Range']}
        for i, box, cid in sorted_boxes:
            text = ocr_text_from_box(image, box)
            label = field_map.get(cid, f"Class {cid}")
            if row_data[label]:
                row_data[label] += " " + text
            else:
                row_data[label] = text
        data_rows.append(row_data)
    return pd.DataFrame(data_rows)

def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = field_map.get(class_ids[i], f"Class {class_ids[i]}")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x, y - 10 if y - 10 > 10 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return image

# Streamlit UI setup
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown("<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> to test and upload from this: "
    "<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True,
)

st.markdown("""
<div style='text-align:center; margin-bottom:0;'>
📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b><br>
<small>📂 Please upload one or more lab report images to start extraction.</small>
</div>
""", unsafe_allow_html=True)

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "extracted_dfs" not in st.session_state:
    st.session_state["extracted_dfs"] = []
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = "file_uploader"

uploaded_files = st.file_uploader(
    " ", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=st.session_state["uploader_key"]
)

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files
    st.session_state["extracted_dfs"] = []

if st.session_state["uploaded_files"]:
    model = load_yolo_model()

    for file in st.session_state["uploaded_files"]:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            with st.spinner("🔍 Running YOLOv5 Detection and EasyOCR..."):
                image = np.array(Image.open(file).convert("RGB"))
                preds, input_img = predict_yolo(model, image)
                indices, boxes, class_ids = process_predictions(preds, input_img)
                if len(indices) == 0:
                    st.warning("⚠️ No fields detected in this image.")
                    continue
                df = extract_rows(image, boxes, class_ids)
                st.session_state["extracted_dfs"].append((file.name, df))

    for fname, df in st.session_state["extracted_dfs"]:
        st.markdown(f"<h5 style='text-align:center;'>✅ Extraction Complete for {fname}!</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        file_obj = next((f for f in st.session_state["uploaded_files"] if f.name == fname), None)
        if file_obj:
            image = np.array(Image.open(file_obj).convert("RGB"))
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)
            img_with_boxes = draw_boxes(image.copy(), boxes, indices, class_ids)
            st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
            st.image(img_with_boxes, use_column_width=True)

    combined_df = pd.concat([df for _, df in st.session_state["extracted_dfs"]], ignore_index=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        c1, c2 = st.columns(2, gap="large")
        csv = combined_df.to_csv(index=False).encode("utf-8")
        with c1:
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="extracted_lab_report.csv",
                mime="text/csv",
            )
        with c2:
            if st.button("🧹 Clear All"):
                st.session_state["uploaded_files"] = []
                st.session_state["extracted_dfs"] = []
                st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
else:
    st.info("Upload one or more lab report images to extract data.")
