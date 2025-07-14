import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
from io import BytesIO

# 🧠 I am mapping detected class IDs to readable field names
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# ✅ I am loading the YOLOv5 ONNX model

def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    # 🧠 I am reading the ONNX model into OpenCV DNN
    model = cv2.dnn.readNetFromONNX(model_path)
    return model

# 🔍 I am running inference with the YOLO model

def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    # 🧠 I am padding the image to make it square for YOLO
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(
        input_img, 1 / 255, (640, 640), swapRB=True, crop=False
    )
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# 📦 I am post‑processing YOLO predictions

def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    detections = preds[0]
    h, w = input_img.shape[:2]
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
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))
                class_ids.append(class_id)

    # 🧠 I am applying Non‑Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes, class_ids

# 🔡 I am extracting text from detected table cells using EasyOCR

def extract_table_text(image, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {key: [] for key in class_map.values()}

    for i in indices:
        if i >= len(boxes) or i >= len(class_ids):
            # 🧠 I am skipping invalid indices
            continue
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")

        # 🧠 I am cropping the detected region
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # 🧠 I am enhancing the crop for better OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            # 🧠 I am reading text lines from the ROI
            lines = reader.readtext(roi, detail=0)
        except Exception:
            lines = []

        for line in lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    # 🧠 I am padding shorter columns so DataFrame creation stays consistent
    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    df = pd.DataFrame(results)
    return df

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

# 💾 I am converting numpy image to bytes so user can download it

def get_image_download_bytes(annotated_np):
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_np, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    annotated_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf

# 🎯 I am setting up the Streamlit UI

st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

query_params = st.query_params
if "clear" in query_params:
    # 🧹 I am resetting the uploader when ?clear=true is present
    uploaded_files = None
    st.query_params.clear()  # I am clearing the query param after use
else:
    uploaded_files = st.file_uploader(
        " ",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

st.markdown("""
<div style='text-align:center;'>
    <h2>🩺🧪 Lab Report OCR Extractor 🧾</h2>
    📥 <b>Download sample Lab Reports (JPG)</b> to test and upload from this:
    <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a><br><br>
    📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b><br>
    <small>📂 Please upload one or more lab report images to start extraction.</small>
</div><br>
""", unsafe_allow_html=True)

clear_trigger = st.button("🧹 Clear All")
if clear_trigger:
    # 🧹 I am triggering upload reset by setting URL query param
    st.query_params.clear()
    st.query_params.update({"clear": "true"})
    st.stop()

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

        annotated_img = draw_boxes(image.copy(), boxes, indices, class_ids)
        img_bytes = get_image_download_bytes(annotated_img)

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        st.image(annotated_img, use_container_width=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            col_dl, col_ann = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{file.name}_ocr.csv",
                    mime="text/csv",
                )
            with col_ann:
                st.download_button(
                    "🖼️ Download Annotated Image",
                    img_bytes,
                    file_name=f"{file.name}_annotated.png",
                    mime="image/png",
                )
