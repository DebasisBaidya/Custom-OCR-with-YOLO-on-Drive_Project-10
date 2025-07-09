import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr
import os
import re

# ✅ Class map from YOLO training
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# ✅ Load YOLO model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model 'best.onnx' not found in current directory.")
        st.stop()
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ Run YOLO forward
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[:h, :w] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_image

# ✅ Process YOLO detections
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

# ✅ OCR + smart row grouping
def extract_clean_rows(image, boxes, indices, class_ids, reader):
    items = []

    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Unknown")
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
                "cy": cy
            })

    # ✅ Group into rows by Y-center proximity
    items.sort(key=lambda x: x["cy"])
    grouped_rows = []
    row_threshold = 35

    for item in items:
        placed = False
        for row in grouped_rows:
            if abs(row["cy"] - item["cy"]) < row_threshold:
                row["items"].append(item)
                row["cy_vals"].append(item["cy"])
                row["cy"] = int(np.mean(row["cy_vals"]))
                placed = True
                break
        if not placed:
            grouped_rows.append({
                "cy": item["cy"],
                "cy_vals": [item["cy"]],
                "items": [item]
            })

    # ✅ Build final rows
    final_data = []
    for row in grouped_rows:
        row_dict = {
            "Test Name": "",
            "Value": "",
            "Units": "",
            "Reference Range": ""
        }
        test_names = [i["text"] for i in row["items"] if i["label"] == "Test Name"]
        row_dict["Test Name"] = " ".join(test_names).strip()

        for i in row["items"]:
            if i["label"] != "Test Name" and not row_dict[i["label"]]:
                row_dict[i["label"]] = i["text"]

        final_data.append(row_dict)

    df = pd.DataFrame(final_data)

    # ✅ Optional: flag abnormal
    def is_abnormal(row):
        try:
            val = float(row["Value"].replace(",", "."))
            nums = re.findall(r"[\d.]+", row["Reference Range"])
            if len(nums) >= 2:
                low, high = float(nums[0]), float(nums[1])
                return not (low <= val <= high)
        except:
            return False
        return False

    df["Abnormal"] = df.apply(is_abnormal, axis=1)
    return df

# ✅ Draw detection boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# ✅ Streamlit App UI
st.set_page_config(layout="wide")
st.title("🧪 Medical Lab Report OCR – Final Clean Row Extractor")

uploaded_files = st.file_uploader("📤 Upload JPG image(s)", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### 🖼️ File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Detecting & Extracting..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_clean_rows(image, boxes, indices, class_ids, reader)

        st.success("✅ Done!")
        st.dataframe(df.drop(columns="Abnormal"))

        st.download_button("📥 Download CSV",
                           df.drop(columns="Abnormal").to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        boxed = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed, caption="📦 Detected Fields", use_container_width=True)
