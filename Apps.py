import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import easyocr
from itertools import zip_longest

# ✅ Class mapping
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# ✅ Load YOLOv5 model
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("Model file 'best.onnx' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ YOLOv5 detection
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Process predictions
def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    h, w = input_img.shape[:2]
    x_factor = w / 640
    y_factor = h / 640

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

# ✅ OCR field-wise with Y info
def extract_fields(image, boxes, indices, class_ids, reader):
    field_data = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    y_positions = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    for i in indices:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i])
        if not label:
            continue

        crop = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(thresh)

        try:
            text = " ".join(reader.readtext(roi, detail=0)).strip()
        except:
            text = ""

        if text:
            field_data[label].append(text)
            y_positions[label].append(y + h // 2)

    # ✅ Smart merge for fragmented test names
    tn_data = list(zip(y_positions["Test Name"], field_data["Test Name"]))
    tn_data.sort()  # by Y

    merged_test_names = []
    skip = False
    for i in range(len(tn_data)):
        if skip:
            skip = False
            continue
        if i + 1 < len(tn_data):
            gap = tn_data[i + 1][0] - tn_data[i][0]
            if gap < 30:
                combined = tn_data[i][1] + " " + tn_data[i + 1][1]
                merged_test_names.append(combined)
                skip = True
            else:
                merged_test_names.append(tn_data[i][1])
        else:
            merged_test_names.append(tn_data[i][1])

    # ✅ Align other fields
    max_len = max(len(merged_test_names),
                  len(field_data["Value"]),
                  len(field_data["Units"]),
                  len(field_data["Reference Range"]))

    def pad(col):
        return col + [""] * (max_len - len(col))

    df = pd.DataFrame({
        "Test Name": pad(merged_test_names),
        "Value": pad(field_data["Value"]),
        "Units": pad(field_data["Units"]),
        "Reference Range": pad(field_data["Reference Range"])
    })

    return df

# ✅ Draw boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Streamlit UI
st.set_page_config(layout="wide")
st.title("🧪 Medical Lab OCR – Clean Rows, Merged Test Names")

uploaded_files = st.file_uploader("📤 Upload JPG image(s)", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### 📄 File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Extracting..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_fields(image, boxes, indices, class_ids, reader)

        st.success("✅ Done!")
        st.dataframe(df)

        st.download_button("📥 Download CSV",
                           df.to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        boxed = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed, caption="📦 Detected Boxes", use_container_width=True)
