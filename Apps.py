# --------------------------------------------------
# 🏗️ I'm importing all required libraries
# --------------------------------------------------
import os, cv2, re, numpy as np, pandas as pd, streamlit as st
from PIL import Image
import easyocr, random, datetime  # (I'm adding datetime to tag debug images)

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
# 🔍 I'm running YOLO prediction on a padded copy
# --------------------------------------------------
def predict_yolo(model, image):
    h, w = image.shape[:2]
    max_side = max(h, w)

    # I'm padding image to square (top‑left anchored)
    padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    padded[:h, :w] = image

    # I'm converting to a blob
    blob = cv2.dnn.blobFromImage(padded, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, padded

# --------------------------------------------------
# 📦 I'm post‑processing YOLO predictions
# --------------------------------------------------
def process_predictions(preds, padded_img, conf_thresh=0.4, score_thresh=0.25):
    boxes, confidences, class_ids = [], [], []
    H, W = padded_img.shape[:2]
    x_factor, y_factor = W / 640, H / 640

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
# 🔡 I'm extracting exactly one line of text per field
# --------------------------------------------------
def extract_table_text(orig_img, boxes, idxs, class_ids, debug=False):
    reader = easyocr.Reader(["en"], gpu=False)
    results = {v: [] for v in class_map.values()}
    field_count = {v: 0 for v in class_map.values()}

    H, W = orig_img.shape[:2]

    for i in idxs:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        field_count[label] += 1  # 🧮 I'm counting detections per class

        # 🛡️ I'm clamping coordinates
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w <= 0 or crop_h <= 0:
            # I'm adding blank to keep row count consistent
            results[label].append("")
            if debug:
                st.warning(f"⚠️ Skipping invalid crop for {label} (size {crop_w}×{crop_h})")
            continue

        crop = orig_img[y1:y2, x1:x2]

        # 🧪 I'm preprocessing for OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.bitwise_not(bin_inv)

        try:
            lines = reader.readtext(roi, detail=0)
        except Exception as e:
            lines = []
            if debug:
                st.error(f"❌ OCR failed for {label}: {e}")

        # I'm using only the first valid line
        clean_lines = [ln.strip() for ln in lines if ln.strip()]
        results[label].append(clean_lines[0] if clean_lines else "")

    # 🗒️ I'm summarising counts
    st.info("📊 Detected count → " + ", ".join(f"{k}:{v}" for k, v in field_count.items()))

    # 🧱 I'm padding rows
    max_len = max(len(v) for v in results.values()) if results else 0
    for k in results:
        results[k] += [""] * (max_len - len(results[k]))

    return pd.DataFrame(results)

# --------------------------------------------------
# 🖼️ I'm drawing boxes on original for debug
# --------------------------------------------------
def draw_boxes_on_original(orig_img, boxes, idxs, class_ids):
    annotated = orig_img.copy()
    H, W = annotated.shape[:2]

    for i in idxs:
        x, y, w, h = boxes[i]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            class_map.get(class_ids[i], "Field"),
            (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return annotated

# --------------------------------------------------
# 🎯 I'm building the Streamlit interface
# --------------------------------------------------
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

# I'm adding a sidebar toggle for debug mode
debug_mode = st.sidebar.checkbox("🔧 Debug mode (show boxes & logs)", value=False)

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
        📤 <b>Upload lab reports (.jpg, .jpeg, or .png)</b><br>
        <small>📂 Please upload one or more images to start extraction.</small>
    </div>
    """,
    unsafe_allow_html=True,
)

# I'm managing uploader key for instant reset
uploader_key = st.session_state.get("uploader_key", "file_uploader_0")
files = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=uploader_key,
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

                # 🖼️ I'm optionally showing debug boxes before OCR
                if debug_mode:
                    st.image(
                        draw_boxes_on_original(orig, boxes, idxs, class_ids),
                        caption="🔧 YOLO detections (debug)",
                        use_container_width=True,
                    )

                df = extract_table_text(orig, boxes, idxs, class_ids, debug=debug_mode)

        # 🧾 I'm displaying extracted table
        st.markdown("<h5 style='text-align:center;'>✅ Extraction Complete!</h5>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>🧾 Extracted Table</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        # 🖼️ I'm always showing final annotated image (for reviewer)
        st.markdown("<h5 style='text-align:center;'>📦 Detected Fields</h5>", unsafe_allow_html=True)
        st.image(draw_boxes_on_original(orig, boxes, idxs, class_ids), use_container_width=True)

        # 📤 I'm adding download and clear buttons
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
                    st.session_state["uploader_key"] = f"file_uploader_{random.randint(1_000_000, 9_999_999)}"
                    st.experimental_rerun()
