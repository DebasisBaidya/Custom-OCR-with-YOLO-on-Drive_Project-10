# --------------------------------------------------
# 🏗️ I'm importing all required libraries
# --------------------------------------------------
import os, cv2, re, numpy as np, pandas as pd, streamlit as st
from PIL import Image
import easyocr

# --------------------------------------------------
# 🧠 I'm defining the class mapping for YOLO labels
# --------------------------------------------------
class_map = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}

# --------------------------------------------------
# 🛠️ I'm loading the YOLOv5 ONNX model
# --------------------------------------------------
def load_yolo_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'best.onnx' not found.")
        st.stop()
    return cv2.dnn.readNetFromONNX(model_path)

# --------------------------------------------------
# 🔍 I'm running YOLO prediction on the input image
# --------------------------------------------------
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_rc = max(h, w)

    # I'm padding to a square canvas (black fill) because YOLO expects 640 × 640
    padded = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    padded[:h, :w] = image

    # I'm preparing the blob for the network
    blob = cv2.dnn.blobFromImage(padded, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, padded  # (I’m returning the padded image so coords stay consistent)

# --------------------------------------------------
# 📦 I'm post‑processing YOLO predictions
# --------------------------------------------------
def process_predictions(preds, padded_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    H, W = padded_img.shape[:2]
    x_factor, y_factor = W / 640, H / 640  # (I’m scaling back from YOLO grid)

    for det in preds[0]:
        conf = det[4]
        if conf > conf_thresh:
            scores = det[5:]
            cls = np.argmax(scores)
            if scores[cls] > score_thresh:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                boxes.append([x, y, int(bw * x_factor), int(bh * y_factor)])
                confidences.append(float(conf))
                class_ids.append(cls)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, 0.45)
    return (idxs.flatten() if len(idxs) else []), boxes, class_ids

# --------------------------------------------------
# 🔡 I'm extracting text for each detected cell
# --------------------------------------------------
def extract_table_text(image, boxes, idxs, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {v: [] for v in class_map.values()}

    for i in idxs:
        if i >= len(boxes):
            continue

        # I'm clamping bbox to image bounds so I never slice out of range
        x, y, w, h = boxes[i]
        H, W = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # I'm pre‑processing the crop for cleaner OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(binary)

        # I'm reading text lines
        try:
            lines = reader.readtext(roi, detail=0)
        except Exception:
            lines = []

        label = class_map.get(class_ids[i], "Field")
        for line in lines:
            clean = line.strip()
            if clean:
                results[label].append(clean)

    # I'm removing duplicates while keeping order so “extra units” vanish
    for k in results:
        seen = set()
        deduped = []
        for item in results[k]:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        results[k] = deduped

    # I'm padding columns so DataFrame stays rectangular
    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    return pd.DataFrame(results)

# --------------------------------------------------
# 🖼️ I'm drawing bboxes on a copy of the *original* image
# --------------------------------------------------
def draw_boxes_on_original(orig_img, boxes, idxs, class_ids):
    annotated = orig_img.copy()
    for i in idxs:
        x, y, w, h = boxes[i]
        x2, y2 = x + w, y + h
        cv2.rectangle(annotated, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            class_map.get(class_ids[i], "Field"),
            (x, y - 10 if y - 10 > 10 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return annotated

# --------------------------------------------------
# 🎯 I'm building the Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown(
    "<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align:center;'>
        📥 <b>Download sample Lab Reports (JPG)</b>:
        <a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a>
    </div><br>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align:center; margin-bottom:0;'>
        📤 <b>Upload lab reports (.jpg, .jpeg, .png)</b><br>
        <small>📂 Please upload one or more images to start extraction.</small>
    </div>
    """,
    unsafe_allow_html=True,
)

uploader_key = st.session_state.get("uploader_key", "file_uploader_0")
files = st.file_uploader(
    " ", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=uploader_key
)

# --------------------------------------------------
# 📑 I'm processing each uploaded image
# --------------------------------------------------
if files:
    model = load_yolo_model()

    for f in files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing: {f.name}</h4>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            with st.spinner("🔍 Running YOLOv5 detection and OCR..."):
                orig = np.array(Image.open(f).convert("RGB"))

                preds, padded = predict_yolo(model, orig)
                idxs, boxes, class_ids = process_predictions(preds, padded)

                if not len(idxs):
                    st.warning("⚠️ No fields detected in this image.")
                    continue

                # I'm extracting text using the *original* image to avoid black borders
                df = extract_table_text(orig, boxes, idxs, class_ids)

        # I'm showing the extracted table
        st.markdown("<h5 style='text-align:center;'>✅ Extraction Complete!</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        # I'm displaying the annotated original image (no black padding)
        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields</h5>", unsafe_allow_html=True)
        st.image(draw_boxes_on_original(orig, boxes, idxs, class_ids), use_container_width=True)

        # I'm adding download + clear buttons
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            col_dl, col_clr = st.columns(2)

            with col_dl:
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{f.name}_ocr.csv",
                    mime="text/csv",
                )

            with col_clr:
                if st.button("🧹 Clear All"):
                    st.session_state["uploaded_files"] = []
                    st.session_state["uploader_key"] = "file_uploader_" + str(np.random.randint(1_000_000))
