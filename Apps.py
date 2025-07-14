import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract as py

# Field mapping
field_map = {0: 'Test Name', 1: 'Value', 2: 'Units', 3: 'Reference Range'}

def load_model(onnx_model_path='best.onnx'):
    if not os.path.exists(onnx_model_path):
        st.error(f"Model file '{onnx_model_path}' not found.")
        st.stop()
    model = cv2.dnn.readNetFromONNX(onnx_model_path)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model

def predict(model, image):
    wh = 640
    h, w, _ = image.shape
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square[:h, :w] = image
    blob = cv2.dnn.blobFromImage(square, 1/255, (wh, wh), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, square

def process(preds, image, conf=0.4, score=0.25):
    boxes, scores, class_ids = [], [], []
    h, w = image.shape[:2]
    x_scale, y_scale = w / 640, h / 640
    for det in preds[0]:
        if det[4] > conf:
            cls_scores = det[5:]
            cls_id = int(np.argmax(cls_scores))
            if cls_scores[cls_id] > score:
                cx, cy, bw, bh = det[0:4]
                x = int((cx - bw/2) * x_scale)
                y = int((cy - bh/2) * y_scale)
                boxes.append([x, y, int(bw * x_scale), int(bh * y_scale)])
                scores.append(float(det[4]))
                class_ids.append(cls_id)
    idx = cv2.dnn.NMSBoxes(boxes, scores, score, 0.45)
    if idx is None or len(idx) == 0:
        return [], boxes, class_ids
    return idx.flatten(), boxes, class_ids

def ocr_text_from_box(image, box):
    x, y, w, h = box
    crop = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = cv2.bitwise_not(th)
    try:
        text = py.image_to_string(roi, config='--oem 3 --psm 6').strip()
    except:
        text = ""
    return text.replace('\n', ' ').strip()

def group_boxes_by_rows(boxes, class_ids, threshold=15):
    # Cluster boxes by their vertical centers to identify rows
    rows = []
    for i, box in enumerate(boxes):
        x, y, w, h = box
        center_y = y + h/2
        placed = False
        for row in rows:
            # Compare with existing row center y
            if abs(row['center_y'] - center_y) < threshold:
                row['boxes'].append((i, box, class_ids[i]))
                # Update row center_y average
                row['center_y'] = np.mean([b[1][1] + b[1][3]/2 for b in row['boxes']])
                placed = True
                break
        if not placed:
            rows.append({'center_y': center_y, 'boxes': [(i, box, class_ids[i])]})
    # Sort rows top to bottom
    rows.sort(key=lambda r: r['center_y'])
    return rows

def extract_rows(image, boxes, class_ids):
    rows = group_boxes_by_rows(boxes, class_ids)
    data_rows = []
    for row in rows:
        # Sort boxes in row left to right
        sorted_boxes = sorted(row['boxes'], key=lambda b: b[1][0])
        row_data = {field: "" for field in field_map.values()}
        for i, box, cid in sorted_boxes:
            text = ocr_text_from_box(image, box)
            label = field_map.get(cid, f"Class {cid}")
            # Append text if multiple boxes for same field in row
            if row_data[label]:
                row_data[label] += " " + text
            else:
                row_data[label] = text
        data_rows.append(row_data)
    return pd.DataFrame(data_rows)

def draw_boxes(image, boxes, idx, class_ids):
    for i in idx:
        x, y, w, h = boxes[i]
        label = field_map.get(class_ids[i], f"Class {class_ids[i]}")
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return image

# Streamlit UI
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")
st.title("🩺🧪 Lab Report OCR Extractor 🧾")

uploaded_file = st.file_uploader("Upload lab report image(s)", type=['jpg','jpeg','png'], accept_multiple_files=True)

if uploaded_file:
    model = load_model()
    all_dfs = []
    for file in uploaded_file:
        st.markdown(f"### Processing: {file.name}")
        img = np.array(Image.open(file).convert("RGB"))
        preds, square_img = predict(model, img)
        idx, boxes, class_ids = process(preds, square_img)
        if len(idx) == 0:
            st.warning("No fields detected.")
            continue
        df = extract_rows(img, boxes, class_ids)
        all_dfs.append(df)
        st.dataframe(df, use_container_width=True)

        img_annotated = draw_boxes(img.copy(), boxes, idx, class_ids)
        st.image(img_annotated, caption="Detected fields with bounding boxes", use_column_width=True)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        csv = combined_df.to_csv(index=False).encode('utf-8')

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.download_button("📥 Download as CSV", csv, "lab_report_extracted.csv", "text/csv")
            with c2:
                if st.button("🧹 Clear All"):
                    st.experimental_rerun()
else:
    st.info("Upload one or more lab report images to extract data.")
