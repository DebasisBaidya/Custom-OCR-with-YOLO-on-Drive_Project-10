import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import os
from pdf2image import convert_from_bytes
import re
from itertools import zip_longest

# ✅ Load model using cv2.dnn.readNet
def load_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ YOLO forward pass
def predict_yolo(model, image):
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_image

# ✅ Parse YOLO output
def process_predictions(preds, input_image, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    detections = preds[0]
    H, W = input_image.shape[:2]
    x_factor = W / 640
    y_factor = H / 640

    for det in detections:
        conf = det[4]
        if conf > conf_thresh:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_thresh:
                cx, cy, w, h = det[:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                boxes.append([left, top, int(w * x_factor), int(h * y_factor)])
                confidences.append(float(conf))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return indices, boxes, class_ids

# ✅ Preprocessing for OCR
def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_not(thresh)

# ✅ Extract and group fields row-wise using label & y-position
def extract_fields_smart(image, boxes, indices, class_ids, reader):
    field_labels = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}
    detections = []

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        crop = image[y:y+h, x:x+w]
        roi = preprocess_crop(crop)
        text = " ".join(reader.readtext(roi, detail=0)).strip()
        if text:
            detections.append({
                "label": field_labels.get(class_ids[i], f"Class {class_ids[i]}"),
                "text": text,
                "x": x,
                "y": y,
                "cx": x + w//2,
                "cy": y + h//2
            })

    detections.sort(key=lambda d: d["cy"])
    rows = []
    row_thresh = 35

    for det in detections:
        placed = False
        for row in rows:
            if abs(row["cy"] - det["cy"]) <= row_thresh:
                row["fields"].append(det)
                row["cy_vals"].append(det["cy"])
                row["cy"] = int(np.mean(row["cy_vals"]))
                placed = True
                break
        if not placed:
            rows.append({"cy": det["cy"], "fields": [det], "cy_vals": [det["cy"]]})

    final_rows = []
    for row in rows:
        grouped = {"Test Name": [], "Value": [], "Units": [], "Reference Range": []}
        for f in row["fields"]:
            if f["label"] in grouped:
                grouped[f["label"]].append(f["text"])
        max_len = max(len(grouped[k]) for k in grouped)
        for row_items in zip_longest(grouped["Test Name"], grouped["Value"], grouped["Units"], grouped["Reference Range"], fillvalue=""):
            final_rows.append({
                "Test Name": row_items[0],
                "Value": row_items[1],
                "Units": row_items[2],
                "Reference Range": row_items[3]
            })

    df = pd.DataFrame(final_rows)

    def is_abnormal(row):
        try:
            val = float(row["Value"].replace(",", "."))
            range_match = re.findall(r"[\d.]+", row["Reference Range"])
            if len(range_match) >= 2:
                low, high = float(range_match[0]), float(range_match[1])
                return not (low <= val <= high)
        except:
            return False
        return False

    df["Abnormal"] = df.apply(is_abnormal, axis=1)
    return df

# ✅ Draw YOLO boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Handle PDFs or images
def pdf_to_images(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        images = convert_from_bytes(uploaded_file.read(), dpi=300)
        return [np.array(img.convert("RGB")) for img in images]
    else:
        return [np.array(Image.open(uploaded_file).convert("RGB"))]

# ✅ Streamlit App
st.set_page_config(layout="wide")
st.title("🧠 Smart Medical Report OCR (YOLOv5 + EasyOCR + Range Check)")

uploaded_files = st.file_uploader("📤 Upload JPG/PNG or PDF", type=["jpg", "png", "pdf"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        pages = pdf_to_images(file)

        for page_num, image in enumerate(pages):
            st.markdown(f"### 📄 File: `{file.name}` - Page {page_num+1}")
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No detections found.")
                continue

            df = extract_fields_smart(image, boxes, indices, class_ids, reader)
            st.success("✅ OCR Complete!")

            def highlight_abnormal(row):
                return ["background-color: #ffdddd" if row.get("Abnormal") else ""] * len(row)

            st.dataframe(df.drop(columns="Abnormal").style.apply(highlight_abnormal, axis=1))

            st.download_button(
                "📥 Download CSV",
                df.drop(columns="Abnormal").to_csv(index=False),
                file_name=f"{file.name}_page{page_num+1}_ocr.csv",
                mime="text/csv"
            )

            boxed_img = draw_boxes(image.copy(), boxes, indices)
            st.image(boxed_img, caption="📦 Detected Fields", use_container_width=True)
