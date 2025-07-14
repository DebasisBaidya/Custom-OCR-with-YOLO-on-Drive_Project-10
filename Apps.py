# --------------------------------------------------
# 🏗️ I'm importing the required libraries
# --------------------------------------------------
import os, cv2, numpy as np, pandas as pd, streamlit as st
from PIL import Image
import easyocr, random, statistics

# --------------------------------------------------
# 🧠 I'm mapping YOLO class IDs to column names
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
# 🔍 I'm making a square copy and running YOLO
# --------------------------------------------------
def predict_yolo(model, image):
    h, w = image.shape[:2]
    m = max(h, w)
    padded = np.zeros((m, m, 3), dtype=np.uint8)
    padded[:h, :w] = image                                  # I'm top‑left anchoring
    blob = cv2.dnn.blobFromImage(padded, 1 / 255, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)
    preds = model.forward()
    return preds, padded

# --------------------------------------------------
# 📦 I'm post‑processing YOLO outputs
# --------------------------------------------------
def process_predictions(preds, padded, conf=0.4, score=0.25):
    boxes, confid, ids = [], [], []
    H, W = padded.shape[:2]
    fx, fy = W / 640, H / 640

    for det in preds[0]:
        if det[4] > conf:
            cls = np.argmax(det[5:])
            if det[5 + cls] > score:
                cx, cy, bw, bh = det[:4]
                x = int((cx - bw / 2) * fx)
                y = int((cy - bh / 2) * fy)
                boxes.append([x, y, int(bw * fx), int(bh * fy)])
                confid.append(float(det[4]))
                ids.append(cls)

    keep = cv2.dnn.NMSBoxes(boxes, confid, score, 0.45)
    return (keep.flatten() if len(keep) else []), boxes, ids

# --------------------------------------------------
# 🔡 I'm doing OCR and building tidy rows
# --------------------------------------------------
def ocr_to_dataframe(orig, boxes, keep, ids):
    reader = easyocr.Reader(["en"], gpu=False)
    H, W = orig.shape[:2]
    cells = []

    # 📸 I'm reading text from every kept box
    for i in keep:
        x, y, w, h = boxes[i]
        label = class_map.get(ids[i], "Field")
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        crop = orig[y1:y2, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
        text = next((t.strip() for t in reader.readtext(gray, detail=0) if t.strip()), "")
        cy = y + h / 2                                           # I'm capturing the vertical centre
        cells.append({"row_y": cy, "label": label, "text": text})

    if not cells:
        return pd.DataFrame(columns=class_map.values())

    # 🗂️ I'm grouping by y‑coordinate with an adaptive threshold
    cells.sort(key=lambda c: c["row_y"])
    heights = [abs(cells[i]["row_y"] - cells[i - 1]["row_y"]) for i in range(1, len(cells))]
    gap_thr = statistics.median(heights) * 0.6 if heights else 20  # I'm adapting the row gap

    rows, current = [], [cells[0]]
    for prev, nxt in zip(cells, cells[1:]):
        if nxt["row_y"] - prev["row_y"] < gap_thr:
            current.append(nxt)
        else:
            rows.append(current)
            current = [nxt]
    rows.append(current)

    # 🧾 I'm turning each grouped row into a dict
    records = []
    for row in rows:
        rec = {v: "" for v in class_map.values()}
        for cell in row:
            rec[cell["label"]] = cell["text"]
        records.append(rec)

    return pd.DataFrame(records)

# --------------------------------------------------
# 🖼️ I'm drawing labelled boxes on the original
# --------------------------------------------------
def draw_boxes(orig, boxes, keep, ids):
    out = orig.copy()
    H, W = out.shape[:2]
    for i in keep:
        x, y, w, h = boxes[i]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            class_map.get(ids[i], "Field"),
            (x1, y1 - 8 if y1 - 8 > 10 else y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )
    return out

# --------------------------------------------------
# 🎯 I'm setting up the Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Lab Report OCR", layout="centered", page_icon="🧾")

st.markdown("<h2 style='text-align:center;'>🩺🧪 Lab Report OCR Extractor 🧾</h2>", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center;'>📤 <b>Upload lab reports (.jpg, .jpeg, .png)</b></div>",
    unsafe_allow_html=True,
)

# 🔄 I'm using a dynamic key so Clear‑All resets instantly
uploader_key = st.session_state.get("uploader_key", "uploader_0")
files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=uploader_key)

# --------------------------------------------------
# 📑 I'm looping through each uploaded image
# --------------------------------------------------
if files:
    model = load_yolo_model()

    for file in files:
        st.markdown(f"<h4 style='text-align:center;'>📄 {file.name}</h4>", unsafe_allow_html=True)

        with st.spinner("🔍 Extracting..."):
            img = np.array(Image.open(file).convert("RGB"))
            preds, padded = predict_yolo(model, img)
            keep, boxes, ids = process_predictions(preds, padded)

            if not keep.size:
                st.warning("⚠️ No fields detected.")
                continue

            df = ocr_to_dataframe(img, boxes, keep, ids)

        # 🖼️ I'm showing the annotated image
        st.image(draw_boxes(img, boxes, keep, ids), use_container_width=True)

        # 🧾 I'm showing the table
        st.dataframe(df, use_container_width=True)

        # --------------------------------------------------
        # 🎛️ I'm centering the buttons nicely
        # --------------------------------------------------
        spacer_left, main, spacer_right = st.columns([1, 2, 1])
        with main:
            dl_col, clr_col = st.columns(2, gap="small")

            with dl_col:
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"{file.name}_ocr.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with clr_col:
                if st.button("🧹 Clear All", use_container_width=True):
                    # I'm changing the uploader key and rerunning for an instant reset
                    st.session_state["uploader_key"] = f"uploader_{random.randint(1_000_000,9_999_999)}"
                    st.experimental_rerun()
