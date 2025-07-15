# --------------------------------------------------
# 🌿 I'm importing the required libraries
# --------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import re

# --------------------------------------------------
# 🧠 I'm defining class mapping for detected fields
# --------------------------------------------------
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# --------------------------------------------------
# 🧠 I'm splitting value+unit if combined
# --------------------------------------------------
_unit_rx = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([^\d\s]+.*)$", re.I)

UNIT_NORMALISE = {
    "g/dl": "g/dL",
    "mg/dl": "mg/dL",
    "mmol/l": "mmol/L",
    "μiu/ml": "µIU/mL",
    "ulu/m": "µIU/mL",
    "ue/dl": "µIU/mL",
    "no/dl": "ng/dL"
}

def _split_value_unit(txt: str):
    m = _unit_rx.match(txt)
    if not m:
        return txt.strip(), ""
    val, unit = m.groups()
    unit = UNIT_NORMALISE.get(unit.lower(), unit)
    return val.strip(), unit.strip()

# --------------------------------------------------
# 🧠 I'm grouping boxes row-wise and extracting class-wise text
# --------------------------------------------------
def extract_table_text(image, boxes, indices, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {key: [] for key in class_map.values()}
    grouped = []

    for i in indices:
        cls = class_ids[i]
        label = class_map.get(cls, "Field")
        x, y, w, h = boxes[i]
        crop = image[y:y+h, x:x+w]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            text = reader.readtext(roi, detail=0)
        except:
            text = []

        text = [t.strip() for t in text if t.strip()]
        if not text:
            continue

        grouped.append({
            "label": label,
            "x": x,
            "y_center": y + h // 2,
            "text": " ".join(text)
        })

    # Group boxes into rows based on y-center proximity
    grouped.sort(key=lambda x: x["y_center"])
    rows = []
    for entry in grouped:
        placed = False
        for row in rows:
            if abs(row[0]["y_center"] - entry["y_center"]) < 15:
                row.append(entry)
                placed = True
                break
        if not placed:
            rows.append([entry])

    for row in rows:
        row.sort(key=lambda x: x["x"])
        fields = {"Test Name": "", "Value": "", "Units": "", "Reference Range": ""}
        for item in row:
            label = item["label"]
            if label in fields:
                fields[label] += (" " + item["text"]).strip()

        val, unit = _split_value_unit(fields["Value"])
        results["Value"].append(val)
        results["Units"].append(unit or fields["Units"])
        results["Test Name"].append(fields["Test Name"])
        results["Reference Range"].append(fields["Reference Range"])

    max_len = max(len(results[k]) for k in results)
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    return pd.DataFrame(results)

# --------------------------------------------------
# 🎯 I'm building the Streamlit app UI
# --------------------------------------------------
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown(
    "<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> 
    to test and upload from this: 
    <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' 
    target='_blank'>Drive Link</a></div><br>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style='text-align:center; margin-bottom:0;'>
📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b><br>
<small>📂 Please upload one or more lab report images to start extraction.</small>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=st.session_state.get("uploader_key", "file_uploader"),
)

if uploaded_files:
    from yolov5_model import load_yolo_model, predict_yolo, process_predictions, draw_boxes
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

        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields on Image</h5>", unsafe_allow_html=True)
        st.image(draw_boxes(image.copy(), boxes, indices, class_ids), use_container_width=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            col_dl, col_rst = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{file.name}_ocr.csv",
                    mime="text/csv",
                )
            with col_rst:
                if st.button("🧹 Clear All"):
                    st.session_state.clear()
                    st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
                    st.rerun()

            # with col_rst:
            #     if st.button("🧹 Clear All"):
            #         st.session_state["uploaded_files"] = []
            #         st.session_state["extracted_dfs"] = []
            #         st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
            #         st.rerun()
