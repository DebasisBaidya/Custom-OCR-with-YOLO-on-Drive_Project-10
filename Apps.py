import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import os

# ✅ Load YOLOv5 model
def load_model():
    model_path = "best.onnx"  # <-- Update if needed
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.stop()
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ Run YOLOv5
def predict_yolo(model, image):
    INPUT_WH_YOLO = 640
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_image

# ✅ Process YOLO predictions
def process_predictions(predictions, input_image, conf_threshold=0.4, score_threshold=0.25):
    boxes = []
    confidences = []
    detections = predictions[0]
    h, w = input_image.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

    for i, det in enumerate(detections):
        confidence = det[4]
        if confidence > conf_threshold:
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > score_threshold:
                cx, cy, bw, bh = det[0:4]
                x = int((cx - 0.5 * bw) * x_factor)
                y = int((cy - 0.5 * bh) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.45)
    return indices, boxes

# ✅ Preprocess crop image
def preprocess_image(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = cv2.bitwise_not(thresh)
    return roi

# ✅ Smart OCR & dynamic row grouping
def smart_ocr(image, boxes, indices, reader):
    # Read all detected regions with EasyOCR
    extracted = []
    for i in indices.flatten():
        if i >= len(boxes): continue
        x, y, w, h = boxes[i]
        x_end, y_end = min(x + w, image.shape[1]), min(y + h, image.shape[0])
        crop = image[y:y_end, x:x_end]
        roi = preprocess_image(crop)
        text = " ".join(reader.readtext(roi, detail=0)).strip()
        if text:
            extracted.append({"y": y, "x": x, "text": text, "box_id": i})

    # Sort vertically (top to bottom), then left to right
    extracted = sorted(extracted, key=lambda b: (b["y"], b["x"]))

    # Group into rows (y proximity)
    rows = []
    row_threshold = 40
    for item in extracted:
        placed = False
        for row in rows:
            if abs(item["y"] - row["y"]) < row_threshold:
                row["fields"].append(item)
                placed = True
                break
        if not placed:
            rows.append({"y": item["y"], "fields": [item]})

    # Build row-wise table
    structured_rows = []
    for row in rows:
        row_data = {"Test Name": "", "Value": "", "Units": "", "Reference Range": ""}
        for field in sorted(row["fields"], key=lambda f: f["x"]):
            t = field["text"].lower()
            if row_data["Test Name"] == "":
                row_data["Test Name"] = field["text"]
            elif any(sym in t for sym in ["mg", "g/dl", "μ", "µ", "ng", "u/l", "/ml", "%"]):
                row_data["Units"] = field["text"]
            elif any(c in t for c in ["-", "to", "~"]) and len(t) < 20:
                row_data["Reference Range"] = field["text"]
            elif row_data["Value"] == "" and t.replace(".", "").isdigit():
                row_data["Value"] = field["text"]
        structured_rows.append(row_data)

    return pd.DataFrame(structured_rows)

# ✅ Draw detection boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Streamlit App
st.set_page_config(layout="wide")
st.title("🧠 Smart OCR for Medical Lab Reports (YOLOv5 + EasyOCR)")

uploaded_files = st.file_uploader("📤 Upload JPG/PNG image(s)", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for uploaded_file in uploaded_files:
        st.markdown(f"### 🔍 Processing: `{uploaded_file.name}`")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        preds, input_img = predict_yolo(model, image)
        indices, boxes = process_predictions(preds, input_img)

        st.code(f"🧩 Box Indices Detected: {indices.flatten().tolist()}")

        if len(indices) == 0:
            st.warning("⚠️ No boxes detected.")
            continue

        df = smart_ocr(image, boxes, indices, reader)
        st.success("✅ OCR Extraction Complete!")
        st.dataframe(df)

        st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        boxed_image = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_image, caption="📦 Detected Regions", use_container_width=True)
