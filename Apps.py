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
# --------------------------------------------------
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# --------------------------------------------------
# 🧠 I'm loading YOLOv5 ONNX model
# --------------------------------------------------
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    return model

# --------------------------------------------------
# 📸 I'm running YOLOv5 detection on input image
# --------------------------------------------------
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# --------------------------------------------------
# 📦 I'm post‑processing YOLO outputs
# --------------------------------------------------
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

# --------------------------------------------------
# 🔡 I'm extracting OCR text for every detected field
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

    # Group boxes by row
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

        results["Test Name"].append(fields["Test Name"])
        results["Value"].append(fields["Value"])
        results["Units"].append(fields["Units"])
        results["Reference Range"].append(fields["Reference Range"])

    max_len = max(len(results[k]) for k in results)
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    return pd.DataFrame(results)

# --------------------------------------------------
# 🖼️ I'm drawing bounding boxes on original image
# --------------------------------------------------
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
