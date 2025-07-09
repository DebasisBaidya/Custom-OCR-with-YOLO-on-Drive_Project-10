# ✅ Grouping Detected Fields and Creating DataFrame
from itertools import zip_longest

def extract_fields_smart(image, boxes, indices, class_ids, reader):
    field_labels = {0: "Test Name", 1: "Value", 2: "Units", 3: "Reference Range"}
    detections = []

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        crop = image[y:y+h, x:x+w]
        roi = preprocess_crop(crop)
        text = " ".join(reader.readtext(roi, detail=0)).strip()
        if text:
            detections.append({
                "label": field_labels.get(class_ids[i], f"Class {class_ids[i]}"),
                "text": text,
                "x": x,
                "y": y,
                "cx": x + w//2,
                "cy": y + h//2
            })

    # ✅ Group into rows using vertical clustering
    detections.sort(key=lambda d: d["cy"])
    rows = []
    row_thresh = 35

    for det in detections:
        placed = False
        for row in rows:
            if abs(row["cy"] - det["cy"]) <= row_thresh:
                row["fields"].append(det)
                row["cy_vals"].append(det["cy"])
                row["cy"] = int(np.mean(row["cy_vals"]))
                placed = True
                break
        if not placed:
            rows.append({"cy": det["cy"], "fields": [det], "cy_vals": [det["cy"]]})

    final_rows = []
    for row in rows:
        grouped = {"Test Name": [], "Value": [], "Units": [], "Reference Range": []}
        for f in row["fields"]:
            if f["label"] in grouped:
                grouped[f["label"]].append(f["text"])
        max_len = max(len(grouped[k]) for k in grouped)
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
