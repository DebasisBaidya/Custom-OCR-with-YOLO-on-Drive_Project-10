import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import os
import re

# ✅ Class labels from YOLO
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# ✅ Load YOLOv5 ONNX model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("Model file 'best.onnx' not found.")
        st.stop()
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ Run YOLO
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[:h, :w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255.0, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Process YOLO outputs
def process_predictions(preds, image, conf_thresh=0.4, score_thresh=0.25):
    h, w = image.shape[:2]
    x_factor, y_factor = w / 640, h / 640

    boxes, confidences, class_ids = [], [], []

    for det in preds[0]:
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

# ✅ OCR & row grouping
def extract_clean_rows(image, boxes, indices, class_ids, reader):
    items = []

    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Unknown")
        cx = x + w // 2
        cy = y + h // 2

        crop = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(threshed)

        text = " ".join(reader.readtext(roi, detail=0)).strip()
        if text:
            items.append({
                "label": label,
                "text": text,
                "cy": cy,
                "cx": cx
            })

    # ✅ Group into rows by cy (tight threshold)
    row_threshold = 15
    items.sort(key=lambda x: x["cy"])
    rows = []

    for item in items:
        placed = False
        for row in rows:
            if abs(row["cy"] - item["cy"]) < row_threshold:
                row["items"].append(item)
                row["cy_vals"].append(item["cy"])
                row["cy"] = int(np.mean(row["cy_vals"]))
                placed = True
                break
        if not placed:
            rows.append({
                "cy": item["cy"],
                "cy_vals": [item["cy"]],
                "items": [item]
            })

    # ✅ Build final table
    final_data = []
    for row in rows:
        fields = {"Test Name": [], "Value": "", "Units": "", "Reference Range": ""}
        for item in sorted(row["items"], key=lambda i: i["cx"]):  # sort left to right
            if item["label"] == "Test Name":
                fields["Test Name"].append(item["text"])
            elif item["label"] in fields and not fields[item["label"]]:
                fields[item["label"]] = item["text"]

        final_data.append({
            "Test Name": " ".join(fields["Test Name"]).strip(),
            "Value": fields["Value"],
            "Units": fields["Units"],
            "Reference Range": fields["Reference Range"]
        })

    df = pd.DataFrame(final_data)

    # ✅ Optional abnormal logic
    def is_abnormal(row):
        try:
            val = float(row["Value"].replace(",", "."))
            parts = re.findall(r"[\d.]+", row["Reference Range"])
            if len(parts) >= 2:
                low, high = float(parts[0]), float(parts[1])
                return not (low <= val <= high)
        except:
            return False
        return False

    df["Abnormal"] = df.apply(is_abnormal, axis=1)
    return df

# ✅ Draw boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Streamlit app
st.set_page_config(layout="wide")
st.title("🧪 Final Medical OCR (Tight Row Fix ✅)")

uploaded_files = st.file_uploader("📤 Upload JPG image(s)", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### 📄 File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Running detection and OCR..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_clean_rows(image, boxes, indices, class_ids, reader)

        st.success("✅ Extraction Complete!")
        st.dataframe(df.drop(columns="Abnormal"))

        st.download_button("📥 Download CSV",
                           df.drop(columns="Abnormal").to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        preview = draw_boxes(image.copy(), boxes, indices)
        st.image(preview, caption="📦 YOLO Boxes", use_container_width=True)
