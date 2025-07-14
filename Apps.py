# --------------------------------------------------
# 🏗️ I'm importing required libraries
# --------------------------------------------------
import os, cv2, re, numpy as np, pandas as pd, streamlit as st
from PIL import Image
import easyocr
import random

# --------------------------------------------------
# 🧠 I'm defining YOLO class mapping
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
# 🔍 I'm running YOLO prediction
# --------------------------------------------------
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)                                  # I'm finding max side

    padded = np.zeros((max_rc, max_rc, 3), np.uint8)    # I'm creating black square
    padded[:h, :w] = image                              # I'm pasting original

    blob = cv2.dnn.blobFromImage(padded, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, padded

# --------------------------------------------------
# 📦 I'm post-processing predictions
# --------------------------------------------------
def process_predictions(preds, padded_img, conf=0.4, score=0.25):
    H, W = padded_img.shape[:2]
    xf, yf = W / 640, H / 640
    boxes, confs, ids = [], [], []

    for det in preds[0]:
        if det[4] > conf:
            cls_scores = det[5:]
            cls = np.argmax(cls_scores)
            if cls_scores[cls] > score:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * xf)
                y = int((cy - bh / 2) * yf)
                boxes.append([x, y, int(bw * xf), int(bh * yf)])
                confs.append(float(det[4]))
                ids.append(cls)

    idxs = cv2.dnn.NMSBoxes(boxes, confs, score, 0.45)
    return (idxs.flatten() if len(idxs) else []), boxes, ids

# --------------------------------------------------
# 🔡 I'm extracting text cell-by-cell
# --------------------------------------------------
def extract_table_text(orig_img, boxes, idxs, ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {v: [] for v in class_map.values()}

    for i in idxs:
        x, y, w, h = boxes[i]
        H, W = orig_img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = orig_img[y1:y2, x1:x2]
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

        label = class_map.get(ids[i], "Field")
        for line in lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    # I'm de-duplicating each column while keeping order
    for k in results:
        seen, dedup = set(), []
        for txt in results[k]:
            if txt not in seen:
                dedup.append(txt)
                seen.add(txt)
        results[k] = dedup

    # I'm setting the expected row count (using longest of Name/Value)
    row_cnt = max(len(results["Test Name"]), len(results["Value"]))
    if row_cnt == 0:
        return pd.DataFrame()  # I'm returning empty if nothing detected

    # I'm trimming/padding every column to row_cnt
    for k in results:
        results[k] = (results[k] + [""] * row_cnt)[:row_cnt]

    return pd.DataFrame(results)

# --------------------------------------------------
# 🖼️ I'm drawing bboxes on original image
# --------------------------------------------------
def draw_boxes(orig_img, boxes, idxs, ids):
    out = orig_img.copy()
    for i in idxs:
        x, y, w, h = boxes[i]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            class_map.get(ids[i], "Field"),
            (x, y - 10 if y - 10 > 10 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return out

# --------------------------------------------------
# 🎯 I'm building Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown("<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align:center;'>
        📥 <b>Download sample Lab Reports (JPG)</b>:
        <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a>
    </div><br>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align:center; margin-bottom:0;'>
        📤 <b>Upload lab reports (.jpg, .jpeg, .png)</b><br>
        <small>📂 Please upload one or more images to start extraction.</small>
    </div>
    """,
    unsafe_allow_html=True,
)

# I'm using one dynamic key so “Clear All” can reset uploader in one click
u_key = st.session_state.get("uploader_key", "file_uploader_0")
files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=u_key)

# --------------------------------------------------
# 📑 I'm processing each uploaded image
# --------------------------------------------------
if files:
    model = load_yolo_model()

    for f in files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing: {f.name}</h4>", unsafe_allow_html=True)

        with st.spinner("🔍 Running YOLOv5 detection and OCR..."):
            orig = np.array(Image.open(f).convert("RGB"))
            preds, padded = predict_yolo(model, orig)
            idxs, boxes, ids = process_predictions(preds, padded)

            if not len(idxs):
                st.warning("⚠️ No fields detected in this image.")
                continue

            df = extract_table_text(orig, boxes, idxs, ids)

        st.markdown("<h5 style='text-align:center;'>✅ Extraction Complete!</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(orig, boxes, idxs, ids), use_container_width=True)

        # I'm letting the reviewer download the CSV
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), f"{f.name}_ocr.csv", "text/csv")

# --------------------------------------------------
# 🧹 I'm adding ONE global “Clear All” button
# --------------------------------------------------
if st.button("🧹 Clear All"):
    # I'm wiping the uploader & rerunning the script
    st.session_state.pop("uploaded_files", None)
    st.session_state["uploader_key"] = f"file_uploader_{random.randint(1, 1_000_000)}"
    st.experimental_rerun()
