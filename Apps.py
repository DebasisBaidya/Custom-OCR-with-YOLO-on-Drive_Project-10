for row in rows:
    grouped = {"Test Name": [], "Value": [], "Units": [], "Reference Range": []}
    for f in row["fields"]:
        if f["label"] in grouped:
            grouped[f["label"]].append(f["text"])
    max_len = max(len(grouped[k]) for k in grouped)
    from itertools import zip_longest
    for row_items in zip_longest(grouped["Test Name"], grouped["Value"], grouped["Units"], grouped["Reference Range"], fillvalue=""):
        final_rows.append({"Test Name": row_items[0], "Value": row_items[1], "Units": row_items[2], "Reference Range": row_items[3]})

    df = pd.DataFrame(final_rows)

    # ✅ Highlight abnormal values
    def is_abnormal(row):
        try:
            val = float(row["Value"].replace(",", "."))
            range_match = re.findall(r"[\d.]+", row["Reference Range"])
            if len(range_match) >= 2:
                low, high = float(range_match[0]), float(range_match[1])
                return not (low <= val <= high)
        except:
            return False
        return False

    df["Abnormal"] = df.apply(is_abnormal, axis=1)
    return df

# ✅ Draw Boxes
def draw_boxes(image, boxes, indices):
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# ✅ Convert PDF or Image

def pdf_to_images(uploaded_pdf):
    if uploaded_pdf.name.lower().endswith(".pdf"):
        images = convert_from_bytes(uploaded_pdf.read(), dpi=300)
        return [np.array(img.convert("RGB")) for img in images]
    else:
        return [np.array(Image.open(uploaded_pdf).convert("RGB"))]

# ✅ Streamlit App
st.set_page_config(layout="wide")
st.title("🧠 Medical Report OCR (YOLOv5 + EasyOCR + Smart Rows + Range Check)")

uploaded_files = st.file_uploader("📤 Upload JPG/PNG or PDF", type=["jpg", "png", "pdf"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    reader = easyocr.Reader(['en'], gpu=False)

    for file in uploaded_files:
        pages = pdf_to_images(file)

        for page_num, image in enumerate(pages):
            st.markdown(f"### 📄 File: `{file.name}` - Page {page_num+1}")
            preds, input_img = predict_yolo(model, image)
            indices, boxes, class_ids = process_predictions(preds, input_img)

            if len(indices) == 0:
                st.warning("⚠️ No detections found.")
                continue

            df = extract_fields_smart(image, boxes, indices, class_ids, reader)
            st.success("✅ OCR Complete!")

            def highlight_abnormal(row):
                return ["background-color: #ffdddd" if row.get("Abnormal") else ""] * len(row)

            st.dataframe(df.drop(columns="Abnormal").style.apply(highlight_abnormal, axis=1))

            st.download_button("📥 Download CSV", df.drop(columns="Abnormal").to_csv(index=False),
                               file_name=f"{file.name}_page{page_num+1}_ocr.csv", mime="text/csv")

            boxed_img = draw_boxes(image.copy(), boxes, indices)
            st.image(boxed_img, caption="📦 Detected Fields", use_container_width=True)
