import os
import re
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

# 🧠 Class Mapping for detected fields
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# Unit correction map (lowercase keys)
unit_correction_map = {
    "mg/dl": "mg/dl",
    "mdl": "mg/dl",
    "mdu": "mg/dl",
    "mgl": "mg/dl",
    "ng/dl": "ng/dl",
    "ngdl": "ng/dl",
    "µg/dl": "µg/dl",
    "pqdl": "µg/dl",
    "ugdl": "µg/dl",
    "ug/dl": "µg/dl",
    "µg/l": "µg/l",
    "ugl": "µg/l",
    "µiu/ml": "µIU/ml",
    "uiu/ml": "µIU/ml",
    "ulu/ml": "µIU/ml",
    "ulu/m": "µIU/ml",
    "uluml": "µIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "miu/ml": "mIU/ml",
    "mIU/ml": "mIU/ml",
    "ululav": "µIU/ml",
}

def normalize_unit_text(text):
    text = text.lower().strip()
    text = text.replace('p', 'µ')
    text = text.replace('q', 'g')
    text = text.replace('u', 'µ')
    text = re.sub(r"[^a-z0-9/µ]", "", text)
    return text

def extract_unit_from_text(text):
    """
    Extract unit substring from an input text using regex.
    Returns the first matched unit substring or None.
    """
    # Regex pattern to capture common units (µ, mg, ng, IU, ml, etc.)
    pattern = re.compile(
        r"(µ?m?iu/ml|mg/dl|ng/dl|µg/dl|µg/l|mg/l|ng/ml|mg/ml|iu/ml|miu/ml|µiu/ml|µg|mg|ng|ml|l|dl)",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if matches:
        # Return the first match normalized
        return normalize_unit_text(matches[0])
    return None

def correct_units_column(df):
    """
    Given a DataFrame with columns 'Units', 'Value', 'Reference Range',
    extract and correct units robustly.
    """
    corrected_units = []
    for idx, row in df.iterrows():
        candidates = []
        # Check Units column
        unit_text = str(row.get("Units", "")).strip()
        if unit_text:
            unit_candidate = extract_unit_from_text(unit_text)
            if unit_candidate:
                candidates.append(unit_candidate)

        # Also check Value and Reference Range columns for possible units
        for col in ["Value", "Reference Range"]:
            val_text = str(row.get(col, "")).strip()
            if val_text:
                unit_candidate = extract_unit_from_text(val_text)
                if unit_candidate:
                    candidates.append(unit_candidate)

        # Pick the best candidate from candidates list
        corrected = None
        for candidate in candidates:
            if candidate in unit_correction_map:
                corrected = unit_correction_map[candidate]
                break
        # If no candidate matched, fallback to the first candidate normalized or empty
        if not corrected and candidates:
            corrected = candidates[0]
        # If still no candidate, empty string
        if not corrected:
            corrected = ""

        corrected_units.append(corrected)
    return corrected_units

# ✅ Load YOLOv5 ONNX model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    return model

# 🔍 Run YOLO prediction
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# 📦 Process predictions
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices.flatten() if len(indices) > 0 else [], boxes, class_ids

# 🔡 OCR + Table Extraction with Smart Units Correction
def extract_table(image, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {key: [] for key in class_map.values()}

    for i in indices:
        if i >= len(boxes) or i >= len(class_ids):
            continue
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i])
        # Safe crop with boundary check
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            lines = reader.readtext(roi, detail=0)
        except Exception:
            lines = []

        for line in lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    # Extract units-like strings from Reference Range to Units (heuristic)
    auto_units = []
    for val in results["Reference Range"]:
        if any(x in val.lower() for x in ["/", "iu", "ml", "µ", "mg", "ug", "mcg"]):
            auto_units.append(val)
    results["Reference Range"] = [t for t in results["Reference Range"] if t not in auto_units]
    results["Units"].extend(auto_units)

    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    return pd.DataFrame(results)

# 🖼️ Draw bounding boxes on image
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

# 🎯 Streamlit UI
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR Extractor</h2>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> to test and upload from this: "
    "<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True,
)
st.markdown("<div style='text-align:center;'>📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    for file in uploaded_files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            with st.spinner("🔍 Running YOLOv5 Detection and OCR..."):
                image = np.array(Image.open(file).convert("RGB"))
                preds, input_img = predict_yolo(model, image)
                indices, boxes, class_ids = process_predictions(preds, input_img)
                if len(indices) == 0:
                    st.warning("⚠️ No fields detected in this image.")
                    continue
                df = extract_table(image, boxes, indices, class_ids)

                # Apply robust unit corrections
                if "Units" in df.columns:
                    df["Units"] = correct_units_column(df)

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
                    "⬇️ Download CSV", df.to_csv(index=False), file_name=f"{file.name}_ocr.csv", mime="text/csv"
                )
            with col_rst:
                if st.button("🔄 Reset All"):
                    st.session_state.clear()
                    st.experimental_rerun()
else:
    st.info("Please upload one or more lab report images to start extraction.")
