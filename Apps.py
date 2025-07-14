# --------------------------------------------------
# 🏗️ I'm importing the required libraries
# --------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

# --------------------------------------------------
# 🧠 I'm defining class mapping for detected fields
#     (0 ➜ Test Name, 1 ➜ Value, 2 ➜ Units, 3 ➜ Reference Range)
# --------------------------------------------------
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range",
}

# --------------------------------------------------
# 🧠 I'm loading YOLOv5 ONNX model
# --------------------------------------------------
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    return cv2.dnn.readNetFromONNX(model_path)

# --------------------------------------------------
# 📸 I'm running YOLOv5 detection on input image
# --------------------------------------------------
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    inp = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    inp[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(inp, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    return model.forward(), inp

# --------------------------------------------------
# 📦 I'm post‑processing YOLO outputs
# --------------------------------------------------
def process_predictions(preds, input_img, conf_th=0.4, score_th=0.25):
    boxes, confs, cids = [], [], []
    h, w = input_img.shape[:2]
    fx, fy = w / 640, h / 640
    for det in preds[0]:
        if det[4] > conf_th:
            cid = np.argmax(det[5:])
            if det[5 + cid] > score_th:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * fx)
                y = int((cy - bh / 2) * fy)
                boxes.append([x, y, int(bw * fx), int(bh * fy)])
                confs.append(float(det[4]))
                cids.append(cid)
    idx = cv2.dnn.NMSBoxes(boxes, confs, score_th, 0.45)
    return idx.flatten() if len(idx) else [], boxes, cids

# --------------------------------------------------
# 🔡 I'm extracting OCR text for every detected field
# --------------------------------------------------
def extract_table_text(img, boxes, idxs, cids):
    reader = easyocr.Reader(["en"], gpu=False)
    # empty columns
    cols = {k: [] for k in ["Test Name", "Value", "Units", "Reference Range"]}

    for i in idxs:
        if i >= len(boxes):
            continue
        x, y, w, h = boxes[i]
        crop = img[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            continue

        label = class_map.get(cids[i], "Field")

        # light pre‑processing
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(
            cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        roi = cv2.bitwise_not(th)

        # OCR – keep every non‑blank line
        try:
            for line in reader.readtext(roi, detail=0):
                text = line.strip()
                if text:
                    cols[label].append(text)
        except Exception:
            pass  # ignore OCR errors

    # align lengths by longest column
    max_len = max(len(v) for v in cols.values()) if cols else 0
    for k, v in cols.items():
        v.extend([""] * (max_len - len(v)))

    return pd.DataFrame(cols)

# --------------------------------------------------
# 🖼️ I'm drawing bounding boxes on original image
# --------------------------------------------------
def draw_boxes(img, boxes, idxs, cids):
    for i in idxs:
        x, y, w, h = boxes[i]
        lbl = class_map.get(cids[i], "Field")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, lbl, (x, y - 10 if y > 20 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
    return img

# --------------------------------------------------
# 🎯 I'm building the Streamlit app UI
# --------------------------------------------------
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown("<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>", unsafe_allow_html=True)

st.markdown(
    """
<div style='text-align:center; margin-bottom:0;'>
📤 <b>Upload lab reports (.jpg, .jpeg, .png)</b><br>
<small>Upload one or more images to start extraction.</small>
</div>
""",
    unsafe_allow_html=True,
)

files = st.file_uploader(" ", ["jpg", "jpeg", "png"], accept_multiple_files=True,
                         key=st.session_state.get("uploader_key", "file_uploader"))

if files:
    yolo = load_yolo_model()
    for f in files:
        st.markdown(f"<h4 style='text-align:center;'>📄 {f.name}</h4>", unsafe_allow_html=True)
        with st.spinner("🔍 Detecting & OCR…"):
            img = np.array(Image.open(f).convert("RGB"))
            p, pad = predict_yolo(yolo, img)
            idx, bxs, cids = process_predictions(p, pad)
            if not idx.size:
                st.warning("⚠️ No fields detected.")
                continue
            df = extract_table_text(img, bxs, idx, cids)

        st.dataframe(df, use_container_width=True)
        st.image(draw_boxes(img.copy(), bxs, idx, cids), use_container_width=True)

        col_dl, col_rst = st.columns(2)
        with col_dl:
            st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                               file_name=f"{f.name}_ocr.csv", mime="text/csv")
        with col_rst:
            if st.button("🧹 Clear All"):
                st.session_state["uploaded_files"] = []
                    st.session_state["extracted_dfs"] = []
                    # Changing uploader key to force reset of widget
                    st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
                    st.rerun()  # 🔁 Reloading the app
