# --------------------------------------------------
# 🏗️ I'm importing all required libraries
# --------------------------------------------------
import os, cv2, numpy as np, pandas as pd, streamlit as st
from PIL import Image
import easyocr, random, itertools

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

    # I'm padding to a square canvas so YOLO sizes stay consistent
    padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    padded[:h, :w] = image

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
# 🔡 I'm extracting one clean line from each bounding box
#     and I'm assembling rows smartly for maximum accuracy
# --------------------------------------------------
def ocr_and_structure(orig_img, boxes, idxs, class_ids):
    reader = easyocr.Reader(["en"], gpu=False)
    entries = []

    H, W = orig_img.shape[:2]

    # 🧾 I'm performing OCR on every kept detection
    for i in idxs:
        x, y, w, h = boxes[i]
        label = class_map.get(class_ids[i], "Field")
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        crop = orig_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # I'm preprocessing for sharper OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        text_lines = reader.readtext(gray, detail=0)
        clean = next((ln.strip() for ln in text_lines if ln.strip()), "")

        # I'm storing center‑y for row clustering
        cy = y + h / 2
        entries.append({"label": label, "text": clean, "cy": cy})

    if not entries:
        return pd.DataFrame(columns=class_map.values())

    # --------------------------------------------------
    # 🗂️ I'm grouping detections into rows by vertical proximity
    # --------------------------------------------------
    entries.sort(key=lambda d: d["cy"])
    rows = []
    for _, group in itertools.groupby(entries, key=lambda d, thresh=20: int(d["cy"] // thresh)):
        rows.append(list(group))

    # I'm converting grouped rows into structured dictionaries
    structured = []
    for row in rows:
        row_dict = {v: "" for v in class_map.values()}
        for cell in row:
            row_dict[cell["label"]] = cell["text"]
        structured.append(row_dict)

    return pd.DataFrame(structured)

# --------------------------------------------------
# 🖼️ I'm drawing boxes on the original image
# --------------------------------------------------
def draw_boxes(orig_img, boxes, idxs, class_ids):
    img = orig_img.copy()
    H, W = img.shape[:2]
    for i in idxs:
        x, y, w, h = boxes[i]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            class_map.get(class_ids[i], "Field"),
            (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return img

# --------------------------------------------------
# 🎯 I'm building the Streamlit interface
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
        📤 <b>Upload lab reports (.jpg, .jpeg, or .png)</b><br>
        <small>📂 Please upload one or more images to start extraction.</small>
    </div>
    """,
    unsafe_allow_html=True,
)

# I'm using a session key so Clear‑All resets instantly
uploader_key = st.session_state.get("uploader_key", "file_uploader_0")
files = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=uploader_key,
)

# --------------------------------------------------
# 📑 I'm processing every uploaded image
# --------------------------------------------------
if files:
    model = load_yolo_model()

    for f in files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing: {f.name}</h4>", unsafe_allow_html=True)

        # I'm reading and analysing the image
        image = np.array(Image.open(f).convert("RGB"))
        preds, padded = predict_yolo(model, image)
        idxs, boxes, class_ids = process_predictions(preds, padded)

        if not len(idxs):
            st.warning("⚠️ No fields detected in this image.")
            continue

        # I'm extracting and structuring text
        df = ocr_and_structure(image, boxes, idxs, class_ids)

        # 🧾 I'm showing the table
        st.markdown("<h5 style='text-align:center;'>✅ Extraction Complete!</h5>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        # 🖼️ I'm showing annotated image
        st.image(draw_boxes(image, boxes, idxs, class_ids), use_container_width=True)

        # 📤 I'm adding download & clear buttons
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
