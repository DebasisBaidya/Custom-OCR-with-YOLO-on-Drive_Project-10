import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import easyocr

# ✅ YOLOv5 Class Mapping
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
    model = cv2.dnn.readNetFromONNX(model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

# ✅ Run YOLOv5 forward pass
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)
    input_img = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_img[0:h, 0:w] = image
    blob = cv2.dnn.blobFromImage(input_img, 1/255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, input_img

# ✅ Process YOLO output
def process_predictions(preds, input_img, conf_thresh=0.4, score_thresh=0.25):
    boxes = []
    confidences = []
    class_ids = []
    h, w = input_img.shape[:2]
    x_factor = w / 640
    y_factor = h / 640
    detections = preds[0]

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

# ✅ Extract + smart merge Test Names
def extract_fields_precise(image, boxes, indices, class_ids, reader):
    fields = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    y_coords = {
        "Test Name": [],
        "Value": [],
        "Units": [],
        "Reference Range": []
    }

    for i in indices:
        if i >= len(boxes) or i >= len(class_ids):
            continue

        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i])
        if not label:
            continue

        crop = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        try:
            text = " ".join(reader.readtext(roi, detail=0)).strip()
        except:
            text = ""

        if text:
            fields[label].append(text)
            y_coords[label].append(y + h // 2)

    # ✅ Smart merge Test Names based on Y proximity
    tn = pd.DataFrame({"text": fields["Test Name"], "y": y_coords["Test Name"]})
    tn = tn.sort_values("y").reset_index(drop=True)

    merged_names = []
    temp = ""
    y_thresh = 20  # adjust as needed
    for i in range(len(tn)):
        if i == 0:
            temp = tn.loc[i, "text"]
        else:
            diff = tn.loc[i, "y"] - tn.loc[i - 1, "y"]
            if diff < y_thresh:
                temp += " " + tn.loc[i, "text"]
            else:
                merged_names.append(temp)
                temp = tn.loc[i, "text"]
    if temp:
        merged_names.append(temp)

    # ✅ Pad other fields to match Test Name length
    max_len = len(merged_names)
    def pad(lst):
        return lst + [""] * (max_len - len(lst))

    result_df = pd.DataFrame({
        "Test Name": merged_names,
        "Value": pad(fields["Value"]),
        "Units": pad(fields["Units"]),
        "Reference Range": pad(fields["Reference Range"])
    })

    return result_df

# ✅ Draw boxes
def draw_boxes(image, boxes, indices):
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Streamlit app
st.set_page_config(layout="wide")
st.title("🧪 Final Medical OCR App (Smart Merge)")

uploaded_files = st.file_uploader("📤 Upload JPG Image(s)", type=["jpg"], accept_multiple_files=True)

if uploaded_files:
    model = load_yolo_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        st.markdown(f"### 📄 File: `{file.name}`")
        image = np.array(Image.open(file).convert("RGB"))

        with st.spinner("🔍 Extracting data..."):
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No fields detected.")
                continue

            df = extract_fields_precise(image, boxes, indices, class_ids, reader)

        st.success("✅ Extraction Complete!")
        st.dataframe(df)

        st.download_button("📥 Download CSV",
                           df.to_csv(index=False),
                           file_name=f"{file.name}_ocr.csv",
                           mime="text/csv")

        boxed_img = draw_boxes(image.copy(), boxes, indices)
        st.image(boxed_img, caption="📦 Detected Fields", use_container_width=True)
