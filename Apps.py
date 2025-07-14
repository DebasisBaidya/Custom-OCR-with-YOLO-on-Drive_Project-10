# --------------------------------------------------
# 🏗️ I'm importing all required libraries
# --------------------------------------------------
import os
import cv2
import re          # (I’m importing just in case future tweaks need regex)
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

# --------------------------------------------------
# 🧠 I'm defining the class mapping for YOLO labels
# --------------------------------------------------
class_map = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}

# --------------------------------------------------
# 🛠️ I'm loading the YOLOv5 ONNX model
# --------------------------------------------------
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    return cv2.dnn.readNetFromONNX(model_path)

# --------------------------------------------------
# 🔍 I'm running YOLO prediction on the input image
# --------------------------------------------------
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)

    # I'm padding to a square so YOLO can work with 640 × 640
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image

    # I'm converting to blob format expected by YOLO
    blob = cv2.dnn.blobFromImage(input_img, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img  # (I’m returning the padded image as well)

# --------------------------------------------------
# 📦 I'm post‑processing YOLO predictions
# --------------------------------------------------
def process_predictions(preds, padded_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    h, w = padded_img.shape[:2]
    x_factor, y_factor = w / 640, h / 640

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
    return indices.flatten() if len(indices) else [], boxes, class_ids

# --------------------------------------------------
# 🔡 I'm extracting text from detected table fields
# --------------------------------------------------
def extract_table_text(base_img, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)  # (I’m creating the EasyOCR reader)
    results = {v: [] for v in class_map.values()}

    for i in indices:
        if i >= len(boxes) or i >= len(class_ids):
            continue

        # I'm grabbing and sanitizing coordinates
        x, y, w, h = boxes[i]
        x1 = max(0, min(base_img.shape[1] - 1, x))
        y1 = max(0, min(base_img.shape[0] - 1, y))
        x2 = max(x1 + 1, min(base_img.shape[1], x + w))
        y2 = max(y1 + 1, min(base_img.shape[0], y + h))

        if x2 <= x1 or y2 <= y1:  # (I’m skipping invalid boxes)
            continue

        crop = base_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # I'm pre‑processing the crop for better OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        # I'm reading text lines from the ROI
        try:
            lines = reader.readtext(roi, detail=0)
        except Exception:
            lines = []

        # I'm storing each cleaned line under the detected label
        label = class_map.get(class_ids[i], "Field")
        for line in lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    # I'm padding columns so DataFrame stays rectangular
    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    return pd.DataFrame(results)

# --------------------------------------------------
# 🖼️ I'm drawing bounding boxes on an image
# --------------------------------------------------
def draw_boxes(img, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x, y - 10 if y - 10 > 10 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return img

# --------------------------------------------------
# 🎯 I'm building the Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown("<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align:center;'>
        📥 <b>Download sample Lab Reports (JPG)</b> from:
        <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a>
    </div><br>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align:center; margin-bottom:0;'>
        📤 <b>Upload lab reports (.jpg, .jpeg, or .png)</b><br>
        <small>📂 Please upload one or more images to start extraction.</small>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="uploaded_files",
)

# --------------------------------------------------
# 📑 I'm processing each uploaded file
# --------------------------------------------------
if uploaded_files:
    model = load_yolo_model()

    for file in uploaded_files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing: {file.name}</h4>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            with st.spinner("🔍 Running YOLOv5 detection and OCR..."):
                # I'm loading the original image
                orig_img = np.array(Image.open(file).convert("RGB"))

                # I'm getting YOLO predictions
                preds, padded_img = predict_yolo(model, orig_img)

                # I'm post‑processing predictions
                indices, boxes, class_ids = process_predictions(preds, padded_img)

                if len(indices) == 0:
                    st.warning("⚠️ No fields detected in this image.")
                    continue

                # I'm extracting table text using the *same padded image* used for boxes
                df = extract_table_text(padded_img, boxes, indices, class_ids)

        # I'm displaying the results
        st.markdown("<h5 style='text-align:center;'>✅ Extraction Complete!</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(padded_img.copy(), boxes, indices, class_ids), use_container_width=True)

        # I'm adding download & clear‑all controls
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            col_dl, col_clr = st.columns(2)

            with col_dl:
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{file.name}_ocr.csv",
                    mime="text/csv",
                )

            with col_clr:
                if st.button("🧹 Clear All"):
                    st.session_state["uploaded_files"] = []
                    st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
