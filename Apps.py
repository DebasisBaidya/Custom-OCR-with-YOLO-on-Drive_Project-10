import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

# 🧠 I am mapping detected class IDs to readable field names
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range",
}

# ✅ I am loading the YOLOv5 ONNX model

def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    return cv2.dnn.readNetFromONNX(model_path)

# 🔍 I am running inference with the YOLO model

def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# 📦 I am post‑processing YOLO predictions

def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    detections = preds[0]
    h, w = input_img.shape[:2]
    x_factor, y_factor = w / 640, h / 640
    for det in detections:
        conf = det[4]
        if conf > conf_thresh:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            if scores[class_id] > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes, class_ids

# 🧰 I am preparing advanced preprocessing to handle hazy / low‑quality crops

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    bin_img = cv2.adaptiveThreshold(
        sharpen,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return cv2.bitwise_not(bin_img)

# 🔡 I am extracting text and confidence using EasyOCR with robust preprocessing

def extract_table_text(image, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {key: [] for key in class_map.values()}
    results["Confidence"] = []

    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        crop = image[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            continue
        roi = preprocess_for_ocr(crop)
        try:
            ocr_results = reader.readtext(roi, detail=1, paragraph=False, decoder="beamsearch")
        except Exception:
            ocr_results = []
        if len(ocr_results) == 0:
            try:
                ocr_results = reader.readtext(crop, detail=1, paragraph=False, decoder="beamsearch")
            except Exception:
                ocr_results = []
        for (_bbox, text, conf) in ocr_results:
            clean = text.strip()
            if clean:
                results[label].append(clean)
                results["Confidence"].append(round(conf * 100, 2))
    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        default_val = 0.0 if k == "Confidence" else ""
        results[k] += [default_val] * (max_len - len(results[k]))
    return pd.DataFrame(results)

# 🖼️ I am drawing bounding boxes and labels on the image

def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
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

# 🎯 I am setting up the Streamlit UI

st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown(
    """
<div style='text-align:center;'>
    <h2>🩺🧪 Lab Report OCR Extractor 🧾</h2>
    📥 <b>Download sample Lab Reports (JPG)</b>:
    <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a><br><br>
</div>
""",
    unsafe_allow_html=True,
)

if "clear" in st.query_params:
    st.query_params.clear()

st.markdown(
    """
<div style='text-align:center; margin-bottom:0;'>
📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b><br>
<small>📂 Please upload one or more lab report images to start extraction.</small>
</div>
""",
    unsafe_allow_html=True,
)

uploader_placeholder = st.empty()

uploaded_files = uploader_placeholder.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="uploaded_files",
)

if uploaded_files:
    model = load_yolo_model()
    for idx, file in enumerate(uploaded_files):
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            with st.spinner("🔍 Running Detection & OCR..."):
                image = np.array(Image.open(file).convert("RGB"))
                preds, inp = predict_yolo(model, image)
                inds, boxes, cls_ids = process_predictions(preds, inp)
                if len(inds) == 0:
                    st.warning("⚠️ No fields detected in this image.")
                    continue
                df = extract_table_text(image, boxes, inds, cls_ids)
        st.success("✅ Extraction Complete!")
        st.subheader("🧾 Extracted Table")
        st.dataframe(df, use_container_width=True)

        annotated = draw_boxes(image.copy(), boxes, inds, cls_ids)
        st.subheader("📦 Detected Fields on Image")
        st.image(annotated, use_container_width=True)

        _, center, _ = st.columns([1, 2, 1])
        with center:
            col_csv, col_clr = st.columns(2)
            with col_csv:
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{file.name}_ocr.csv",
                    mime="text/csv",
                    key=f"csv_{idx}",
                )
            with col_clr:
                if st.button("🧹 Clear All", key=f"clear_{idx}"):
                    st.query_params.update({"clear": "true"})
                    st.stop()
