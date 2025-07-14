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
# 🧠 I'm defining class‑id ➜ field mapping
# --------------------------------------------------
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range",
}

# --------------------------------------------------
# 🧰 I'm compiling a quick regex that “looks like a unit”
#      (contains "/" or IU/Iu/iu or ends with mL, L, g)
# --------------------------------------------------
unit_like_rx = re.compile(r"(g/|mg/|mmol/|iu|µiu|ml\b|l\b|/L|/dL)", re.I)

# --------------------------------------------------
# 🧠 I'm loading YOLOv5 ONNX model
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    net = cv2.dnn.readNetFromONNX("best.onnx")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# --------------------------------------------------
# 🧠 I'm loading EasyOCR once per session
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_easyocr():
    return easyocr.Reader(["en"], gpu=False)

# --------------------------------------------------
# 📸 I'm running YOLOv5 detection on input image
# --------------------------------------------------
def predict_yolo(net, img, inp_wh=640):
    h, w, _ = img.shape
    s = max(h, w)
    padded = np.zeros((s, s, 3), dtype=np.uint8)
    padded[0:h, 0:w] = img
    blob = cv2.dnn.blobFromImage(padded, 1/255, (inp_wh, inp_wh), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward(), padded

# --------------------------------------------------
# 📦 I'm post‑processing YOLO outputs
# --------------------------------------------------
def process(preds, padded, conf=0.4, cls_th=0.25):
    boxes, scores, cls_ids = [], [], []
    H, W = padded.shape[:2]
    fx, fy = W/640, H/640
    for det in preds[0]:
        if det[4] < conf:
            continue
        cls_conf = det[5:]
        cls_id = int(np.argmax(cls_conf))
        if cls_conf[cls_id] < cls_th:
            continue
        cx, cy, bw, bh = det[:4]
        x = int((cx - bw/2) * fx)
        y = int((cy - bh/2) * fy)
        boxes.append([x, y, int(bw*fx), int(bh*fy)])
        scores.append(float(det[4]))
        cls_ids.append(cls_id)
    keep = cv2.dnn.NMSBoxes(boxes, scores, cls_th, 0.45)
    return keep.flatten() if len(keep) else [], boxes, cls_ids

# --------------------------------------------------
# 🔡 I'm extracting OCR text for every detected field
#     + smart post‑step that moves unit‑like strings
#       mistakenly classified as "Reference Range"
# --------------------------------------------------
def ocr_fields(img, boxes, keep, cls_ids, reader):
    data = {v: [] for v in class_map.values()}

    for idx in keep:
        cls_id = cls_ids[idx]
        if cls_id not in class_map:
            continue  # skip unknown class ids
        label = class_map[cls_id]

        x, y, w, h = boxes[idx]
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        # minimal preprocessing
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(cv2.GaussianBlur(g, (5,5), 0), 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(th)

        lines = reader.readtext(roi, detail=0) or []
        for line in map(str.strip, lines):
            if line:
                data[label].append(line)

    # ✅ smart units correction: move unit‑looking text
    moved_units = []
    for txt in data["Reference Range"]:
        if unit_like_rx.search(txt):
            moved_units.append(txt)
    if moved_units:
        data["Reference Range"] = [t for t in data["Reference Range"] if t not in moved_units]
        data["Units"].extend(moved_units)

    # pad columns equally
    max_len = max(len(v) for v in data.values())
    for k in data:
        data[k] += [""] * (max_len - len(data[k]))
    return pd.DataFrame(data)

# --------------------------------------------------
# 🖼️ I'm drawing bounding boxes on original image
# --------------------------------------------------
def draw_boxes(img, boxes, keep, cls_ids):
    for idx in keep:
        if cls_ids[idx] not in class_map:
            continue
        x, y, w, h = boxes[idx]
        label = class_map[cls_ids[idx]]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
    return img

# --------------------------------------------------
# 🎯 I'm building the Streamlit UI
# --------------------------------------------------
st.set_page_config("Lab Report OCR", "centered", "🧾")
st.markdown("<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> "
            "from: <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' "
            "target='_blank'>Drive Link</a></div><br>", unsafe_allow_html=True)
st.markdown("""<div style='text-align:center; margin-bottom:0;'>
📤 <b>Upload lab reports (.jpg, .jpeg, .png)</b><br>
<small>📂 Upload one or more images to start extraction.</small>
</div>""", unsafe_allow_html=True)

files = st.file_uploader(" ", ["jpg","jpeg","png"], True,
                         key=st.session_state.get("uploader_key","file_uploader"))

if files:
    net = load_yolo_model()
    reader = load_easyocr()
    for f in files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing : {f.name}</h4>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1,2,1])
        with mid:
            with st.spinner("🔍 Detecting & extracting..."):
                img = np.array(Image.open(f).convert("RGB"))
                preds, padded = predict_yolo(net, img)
                keep, boxes, cls_ids = process(preds, padded)
                if not keep.size:
                    st.warning("⚠️ No fields detected.")
                    continue
                df = ocr_fields(img, boxes, keep, cls_ids, reader)

        st.success("✅ Extraction Complete!")
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(img.copy(), boxes, keep, cls_ids), use_column_width=True)

        _, mid, _ = st.columns([1,2,1])
        with mid:
            dl, clr = st.columns(2)
            with dl:
                st.download_button("⬇️ Download CSV",
                                   df.to_csv(index=False),
                                   f"{f.name}_ocr.csv",
                                   "text/csv")
            with clr:
                
                # --------------------------------------------------
                # 🧹 I'm clearing all session data & rerunning app
                # --------------------------------------------------
                if st.button("🧹 Clear All"):
                    st.session_state["uploaded_files"] = []
                    st.session_state["extracted_dfs"] = []
                    # Changing uploader key to force reset of widget
                    st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
                    st.rerun()  # 🔁 Reloading the app
