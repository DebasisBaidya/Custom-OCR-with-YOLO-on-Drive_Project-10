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

# 🔡 I am extracting text and confidence using EasyOCR

def extract_table_text(image, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {key: [] for key in class_map.values()}
    results["Confidence"] = []
    for i in indices:
        if i >= len(boxes):
            continue
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        crop = image[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)
        try:
            for bbox, text, conf in reader.readtext(roi):
                clean = text.strip()
                if clean:
                    results[label].append(clean)
                    results["Confidence"].append(round(conf * 100, 2))
        except Exception:
            pass
    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        pad_val = 0.0 if k == "Confidence" else ""
        results[k] += [pad_val] * (max_len - len(results[k]))
    return pd.DataFrame(results)

# 🖼️ I am drawing bounding boxes and labels on the image

def draw_boxes(image, boxes, indices, class_ids):
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10 if y - 10 > 10 else y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return image

# 💾 I am converting numpy image to bytes so user can download it

def get_image_download_bytes(img_np):
    pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

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

# 👉 I am handling reset via URL param
if "clear" in st.query_params:
    st.query_params.clear()

# 🧰 I am preparing a placeholder so uploader stays at the bottom
uploader_placeholder = st.empty()

# 📂 I am reading uploaded files (if any)
uploaded_files = uploader_placeholder.file_uploader(
    "📤 Upload lab reports (.jpg, .jpeg, or .png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    model = load_yolo_model()
    for idx, file in enumerate(uploaded_files):
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>", unsafe_allow_html=True)
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
        img_bytes = get_image_download_bytes(annotated)
        st.subheader("📦 Detected Fields on Image")
        st.image(annotated, use_container_width=True)

        # 🎚 I am grouping action buttons together (CSV, Image, Clear)
        _, center, _ = st.columns([1, 2, 1])
        with center:
            col_csv, col_img, col_clr = st.columns(3)
            with col_csv:
                st.download_button("⬇️ CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv", key=f"csv_{idx}")
            with col_img:
                st.download_button("🖼️ Annotated", img_bytes, file_name=f"{file.name}_annotated.png", mime="image/png", key=f"img_{idx}")
            with col_clr:
                if st.button("🧹 Clear", key=f"clear_{idx}"):
                    st.query_params.update({"clear": "true"})
                    st.stop()

