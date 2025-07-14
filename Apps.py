import os
import re
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import easyocr

def normalize_unit_text(text):
    text = text.lower().strip()
    text = text.replace('p', 'µ')
    text = text.replace('q', 'g')
    text = text.replace('u', 'µ')
    text = re.sub(r"[^a-z0-9/µ]", "", text)
    return text

def extract_units_from_texts(texts):
    """
    Extract unit-like substrings from all OCR text lines,
    count their occurrences, and return sorted list with confidence %.
    """
    unit_pattern = re.compile(
        r"(µ?m?iu/ml|mg/dl|ng/dl|µg/dl|µg/l|mg/l|ng/ml|mg/ml|iu/ml|miu/ml|µiu/ml|µg|mg|ng|ml|l|dl)",
        re.IGNORECASE,
    )
    freq = {}
    total_lines = len(texts) if texts else 1
    for text in texts:
        matches = unit_pattern.findall(text)
        for match in matches:
            normalized = normalize_unit_text(match)
            freq[normalized] = freq.get(normalized, 0) + 1

    # Convert counts to percentage confidence (relative frequency)
    units_confidence = []
    for unit, count in freq.items():
        confidence = (count / total_lines) * 100
        units_confidence.append((unit, confidence))
    # Sort descending by confidence
    units_confidence.sort(key=lambda x: x[1], reverse=True)
    return units_confidence

# 🖼️ OCR full image text extraction
def ocr_full_image(image):
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(image, detail=0)
    return results

# 🎯 Streamlit UI
st.set_page_config(page_title="Lab Report OCR Simple Unit Extraction", layout="centered", page_icon="🧾")

st.markdown("<h2 style='text-align:center;'>🧾 Lab Report OCR - Simple Unit Extraction</h2>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;'>📥 <b>Download sample Lab Reports (JPG)</b> to test and upload from this: "
    "<a href='https://drive.google.com/drive/folders/1zgCl1A3HIqOIzgkBrWUFRhVV0dJZsCXC?usp=sharing' target='_blank'>Drive Link</a></div><br>",
    unsafe_allow_html=True,
)
st.markdown("<div style='text-align:center;'>📤 <b>Upload lab reports (.jpg, .jpeg, or .png format)</b></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(" ", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.markdown(f"<h4 style='text-align:center;'>📄 Processing File: {file.name}</h4>", unsafe_allow_html=True)

        with st.spinner("🔍 Running OCR on full image..."):
            image = np.array(Image.open(file).convert("RGB"))
            ocr_texts = ocr_full_image(image)

        st.markdown("<h5 style='text-align:center;'>✅ OCR Text Lines Extracted</h5>", unsafe_allow_html=True)
        st.write(ocr_texts)

        units_confidence = extract_units_from_texts(ocr_texts)

        if units_confidence:
            st.markdown("<h5 style='text-align:center;'>🔎 Extracted Units with Confidence (%)</h5>", unsafe_allow_html=True)
            for unit, conf in units_confidence:
                st.markdown(f"**{unit}** — {conf:.1f}%")
        else:
            st.markdown("<h5 style='text-align:center;'>No units detected.</h5>", unsafe_allow_html=True)

        # Show uploaded image for reference
        st.image(image, caption="Uploaded Lab Report Image", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Reset All"):
                st.session_state.clear()
                st.experimental_rerun()
else:
    st.info("Please upload one or more lab report images to start extraction.")
