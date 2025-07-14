# --------------------------------------------------
# 🏗️ I'm importing the required libraries
# --------------------------------------------------
import os
import cv2
import re
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

# --------------------------------------------------
# 🧠 I'm defining class‑ID ➜ field mapping
# --------------------------------------------------
CLASS_MAP = {
    61: "Test Name",
    14: "Value",
    26: "Units",
    41: "Reference Range",
}

# --------------------------------------------------
# 🔍 I'm preparing helpers to split mixed value‑unit strings
#     e.g. “13.5g/dL” ➜ value = 13.5, unit = g/dL
# --------------------------------------------------
_UNIT_RX = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([^\d\s]+.*)$", re.I)
UNIT_NORMALISE = {
    "g/dl":   "g/dL",
    "mg/dl":  "mg/dL",
    "mmol/l": "mmol/L",
    "μiu/ml": "µIU/mL",
}

def split_value_unit(txt: str):
    """🧠 I'm returning (value, unit); blank unit if none found."""
    m = _UNIT_RX.match(txt)
    if not m:
        return txt.strip(), ""
    val, unit = m.groups()
    unit = UNIT_NORMALISE.get(unit.lower(), unit)
    return val.strip(), unit.strip()

# --------------------------------------------------
# 🧠 I'm loading YOLOv5 ONNX model (cached per session)
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    net = cv2.dnn.readNetFromONNX("Models/best.onnx")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# --------------------------------------------------
# 🧠 I'm loading the EasyOCR reader (cached per session)
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_easyocr_reader():
    return easyocr.Reader(["en"], gpu=False)

# --------------------------------------------------
# 📸 I'm running a forward pass through YOLOv5
# --------------------------------------------------
def predict_yolo(net, image, inp_size=640):
    h, w, _ = image.shape
    s = max(h, w)
    padded = np.zeros((s, s, 3), np.uint8)
    padded[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(padded, 1 / 255, (inp_size, inp_size), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds, padded

# --------------------------------------------------
# 📦 I'm post‑processing YOLO detections (NMS + scaling)
# --------------------------------------------------
def process_predictions(preds, padded, conf_th=0.4, cls_th=0.25):
    boxes, scores, cls_ids = [], [], []
    H, W = padded.shape[:2]
    fx, fy = W / 640, H / 640

    for det in preds[0]:
        obj_conf = det[4]
        if obj_conf < conf_th:
            continue
        cls_conf = det[5:]
        cls_id = int(np.argmax(cls_conf))
        if cls_conf[cls_id] < cls_th:
            continue
        cx, cy, bw, bh = det[:4]
        x = int((cx - bw / 2) * fx)
        y = int((cy - bh / 2) * fy)
        boxes.append([x, y, int(bw * fx), int(bh * fy)])
        scores.append(float(obj_conf))
        cls_ids.append(cls_id)

    keep = cv2.dnn.NMSBoxes(boxes, scores, cls_th, 0.45).flatten() if boxes else []
    return keep, boxes, cls_ids

# --------------------------------------------------
# 🔡 I'm extracting OCR text from each detected field
# --------------------------------------------------
def ocr_from_boxes(image, boxes, keep, cls_ids, reader):
    results = {name: [] for name in CLASS_MAP.values()}

    for idx in keep:
        x, y, w, h = boxes[idx]
        x, y = max(0, x), max(0, y)
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        # 🧰 I'm lightly preprocessing the crop for better OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(th)

        # 🔍 I'm running EasyOCR
        text_lines = reader.readtext(roi, detail=0, paragraph=False)
        field = CLASS_MAP.get(cls_ids[idx], "Unknown")

        for line in map(str.strip, text_lines):
            if not line:
                continue

            # 🧠 I'm fixing value+unit mixing
            if field == "Value":
                val, unit = split_value_unit(line)
                results["Value"].append(val)
                if unit:
                    results["Units"].append(unit)
            else:
                results[field].append(line)

    # 🧱 I'm padding all columns to equal length
    max_len = max(len(v) for v in results.values())
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    return pd.DataFrame(results)

# --------------------------------------------------
# 🖼️ I'm drawing bounding boxes with labels
# --------------------------------------------------
def draw_boxes(img, boxes, keep, cls_ids):
    for idx in keep:
        x, y, w, h = boxes[idx]
        label = CLASS_MAP.get(cls_ids[idx], "?")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img

# --------------------------------------------------
# 🎯 I'm building the Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="🧾 Lab Report OCR", layout="centered", page_icon="🩺")

# 👉 Title
st.markdown(
    "<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>",
    unsafe_allow_html=True,
)

# 👉 Sample link
st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> "
    "from this <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' "
    "target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True,
)

# 👉 Uploader instructions
st.markdown(
    """
<div style='text-align:center; margin-bottom:0;'>
📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b><br>
<small>📂 Please upload one or more lab report images to start extraction.</small>
</div>
""",
    unsafe_allow_html=True,
)

# 👉 File uploader
uploaded_files = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=st.session_state.get("uploader_key", "file_uploader"),
)

# --------------------------------------------------
# 🚀 I'm performing detection & OCR for each image
# --------------------------------------------------
if uploaded_files:
    yolo_net = load_yolo_model()
    ocr_reader = load_easyocr_reader()

    for file in uploaded_files:
        st.markdown(
            f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>",
            unsafe_allow_html=True,
        )

        # 👉 Centered spinner
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            with st.spinner("🔍 Running YOLOv5 Detection and OCR..."):
                img = np.array(Image.open(file).convert("RGB"))

                # 🔎 Detection
                preds, padded = predict_yolo(yolo_net, img)
                keep, boxes, cls_ids = process_predictions(preds, padded)

                if not len(keep):
                    st.warning("⚠️ No fields detected in this image.")
                    continue

                # 🧠 OCR + post‑processing
                df = ocr_from_boxes(img, boxes, keep, cls_ids, ocr_reader)

        # ✅ Completion notice
        st.markdown(
            "<h5 style='text-align:center;'>✅ Extraction Complete!</h5>",
            unsafe_allow_html=True,
        )

        # 🧾 Extracted table
        st.markdown(
            "<h5 style='text-align:center;'>🧾 Extracted Table</h5>",
            unsafe_allow_html=True,
        )
        st.dataframe(df, use_container_width=True)

        # 📦 Detected image with boxes
        st.markdown(
            "<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>",
            unsafe_allow_html=True,
        )
        st.image(draw_boxes(img.copy(), boxes, keep, cls_ids), use_column_width=True)

        # 💾 Download CSV + Clear All (center‑aligned buttons)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            btn_dl, btn_cl = st.columns(2)
            with btn_dl:
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{file.name}_ocr.csv",
                    mime="text/csv",
                )
            with btn_cl:
                if st.button("🧹 Clear All"):
                    st.session_state["uploaded_files"] = []
                    st.session_state["extracted_dfs"] = []
                    # Changing uploader key to force reset of widget
                    st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
                    st.rerun()  # 🔁 Reloading the app
