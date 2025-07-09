# ✅ Streamlit App for Project 10 - Custom OCR (YOLOv5 + EasyOCR)
# Purpose: Detect and structure medical lab report fields using YOLOv5 and EasyOCR

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import easyocr
from PIL import Image
import os

# ✅ Load YOLOv5 model
def load_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# ✅ Predict using YOLOv5 model
def detect_regions(net, image):
    H, W = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()[0]

    boxes, scores, class_ids = [], [], []
    for i in range(output.shape[0]):
        row = output[i]
        conf = row[4]
        if conf >= 0.4:
            scores_all = row[5:]
            class_id = np.argmax(scores_all)
            if scores_all[class_id] > 0.25:
                cx, cy, w, h = row[0:4]
                x = int((cx - w / 2) * W / 640)
                y = int((cy - h / 2) * H / 640)
                width = int(w * W / 640)
                height = int(h * H / 640)
                boxes.append([x, y, width, height])
                scores.append(float(conf))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
    return [boxes[i[0]] for i in indices], [class_ids[i[0]] for i in indices]

# ✅ Map class ids to names
class_map = {
    0: "Test Name",
    1: "Value",
    2: "Units",
    3: "Reference Range"
}

# ✅ Run OCR on each region
def run_ocr(image, boxes, class_ids, reader):
    data = {"Test Name": [], "Value": [], "Units": [], "Reference Range": []}
    for (box, cls_id) in zip(boxes, class_ids):
        x, y, w, h = box
        crop = image[y:y+h, x:x+w]
        result = reader.readtext(crop, detail=0)
        text = " ".join(result).strip()
        label = class_map.get(cls_id, "Unknown")
        data[label].append(text)
    return data

# ✅ Display image with boxes
def draw_boxes(image, boxes, class_ids):
    for (box, cls_id) in zip(boxes, class_ids):
        x, y, w, h = box
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(image, class_map.get(cls_id, ""), (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    return image

# ✅ Streamlit UI
st.set_page_config(layout="wide")
st.title("🩺 Medical Lab Report OCR with YOLOv5 + EasyOCR")
uploaded_files = st.file_uploader("Upload JPG image(s)", type="jpg", accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    reader = easyocr.Reader(["en"], gpu=False)

    for uploaded_file in uploaded_files:
        st.markdown(f"### Processing: {uploaded_file.name}")
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        boxes, class_ids = detect_regions(model, image)
        ocr_result = run_ocr(image, boxes, class_ids, reader)

        # Pad to max length
        max_len = max(len(v) for v in ocr_result.values())
        for k in ocr_result:
            while len(ocr_result[k]) < max_len:
                ocr_result[k].append("")

        df = pd.DataFrame(ocr_result)
        st.dataframe(df)

        st.download_button("📥 Download CSV", df.to_csv(index=False), f"{uploaded_file.name}_ocr.csv", mime="text/csv")

        boxed_img = draw_boxes(image.copy(), boxes, class_ids)
        st.image(boxed_img, caption="Detected Regions", use_container_width=True)
