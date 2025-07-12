# 🧾 Custom Object Character Recognition (OCR) on Google Drive

<p align="center">
  <img src="https://img.shields.io/badge/OpenCV-Used-007ACC?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/numpy-Used-4D77CF?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Used-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTesseract-Used-F89820?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/EasyOCR-Used-FF6600?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TesseractOCR-Used-525252?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv3-Used-28A745?logo=yolo&logoColor=white" />
  <img src="https://img.shields.io/badge/ONNX-Used-9058B4?logo=onnx&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Used-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/PIL-Used-5C3EE8?logo=python&logoColor=white" />
</p>

Welcome to my **Medical Lab Report OCR Project**! 🧪

This project tackles a very specific problem in medical documentation: extracting structured data like **Test Names**, **Values**, **Units**, and **Reference Ranges** from unstructured lab report images.

Manually entering such data is slow, repetitive, and error-prone. So I designed an intelligent solution combining object detection (YOLOv3) with OCR (Tesseract/EasyOCR) to automate this workflow. 🧠🖼️

This is not just a proof of concept — it's a fully working pipeline with model training, preprocessing, post-processing, UI deployment, and CSV export.

> 🔗 [🌐 Streamlit App](https://your-streamlit-app-url) | 🎥 [📽 Demo Video](https://your-demo-video-link)

---

## 🎯 Objective

Build an end-to-end system to:

* 🎯 **Detect**: Identify specific regions in lab reports using YOLOv3
* 📸 **Preprocess**: Clean and prepare those image snippets
* 🧾 **Extract**: Use Tesseract/EasyOCR to pull out actual text from regions
* 🧠 **Structure**: Organize results into a tabular format
* 📤 **Export**: Save as downloadable CSV for use in Excel or EMR systems
* 💻 **Deploy**: Provide a UI via Streamlit so others can test it easily

---

## 🧠 Problem Background

In real-world clinics and diagnostic labs, lab reports are often scanned or saved as JPGs with varied formats. Extracting key medical parameters is essential but mostly done manually.

This project solves that by:

* 🔍 Detecting predefined regions using a trained YOLOv3 model
* 🧾 Reading those regions via OCR (Tesseract/EasyOCR)
* 📊 Structuring the extracted text for automated storage and analysis

---

## 📂 Approach Overview

The pipeline follows these logical steps:

1. 📌 **Image Labeling** – using `LabelImg` for bounding boxes
2. 📁 **Annotation Conversion** – XML to YOLO txt via custom Python script
3. 🧠 **YOLOv3 Training** – on Google Colab (GPU enabled)
4. 🔄 **Model Export** – save best weights to ONNX
5. 🔍 **Detection + Preprocessing** – load model, preprocess with OpenCV
6. 🧾 **Text Recognition** – via EasyOCR or Pytesseract
7. 🧠 **Smart Postprocessing** – e.g., merging multi-word test names
8. 📤 **CSV + Annotated Image Output**
9. 🌐 **Deployment** – Streamlit-based web interface
10. 🎥 **Documentation + Demo** – screenshots, videos for review

---

## 🔧 Features Implemented

* ✅ YOLOv3-based detection (ONNX)
* ✅ Custom image preprocessing: resize, grayscale, blur, threshold
* ✅ OCR with fallback: EasyOCR or Pytesseract
* ✅ Smart merge logic for split fields
* ✅ Streamlit UI with image upload, download buttons
* ✅ Annotated image preview + CSV export

---

## 📁 File Structure

```
📂 project_root
├── 📒 P10.ipynb                # Notebook for model training & XML parsing
├── 📄 Apps.py                 # Streamlit app for UI
├── 🧾 data.yaml              # YOLO label map config
├── 🧠 model/best.onnx        # Trained YOLO model (ONNX format)
├── 📷 data_images/           # Labeled input images
├── 📄 Extract Text from XML.ipynb  # Converts XML → YOLO format
├── 📸 screenshots/            # Add your screenshots here
└── 📄 Instructions.pdf        # Project outline
```

---

## 🖼️ Streamlit Interface (UI Preview)

| Upload & Annotate                               | Extracted Table & CSV                          | Preview Annotated Output                         |
| ----------------------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| ![streamlit1](screenshots/streamlit_upload.jpg) | ![streamlit2](screenshots/streamlit_table.jpg) | ![streamlit3](screenshots/streamlit_preview.jpg) |

> 🔗 Try the app: [🌐 Streamlit Live](https://your-streamlit-app-url)
> 🎬 Watch Demo: [📽 YouTube Video](https://your-demo-video-link)

---

## 📊 Results

* 📌 **Accuracy**: High mAP\@0.5 for bounding box detection
* 🧠 **OCR Success Rate**: Robust across different font sizes/styles
* 🔄 **Smart Field Merging**: e.g., joins "Total" + "Bilirubin"
* 📈 **Usability**: Streamlit UI makes it plug-and-play for any JPG report

---

## 🧪 Sample Output

| Input Image                     | YOLO Detection                     | OCR Output                        |
| ------------------------------- | ---------------------------------- | --------------------------------- |
| ![input](screenshots/input.jpg) | ![yolo](screenshots/detection.jpg) | ![output](screenshots/output.jpg) |

> 📦 Exports: `ocr_result.csv`, `annotated_image.jpg`

---

## 🙋‍♂️ About Me

Hi, I’m **Debasis Baidya** from Kolkata 👋
With **11+ years** of experience in the MIS domain, I am now transitioning into the world of **Data Science**.

* Currently Working as Senior MIS Analyst | Data Science Intern

✅ 80%+ automation of manual processes at my workplace
📊 Skilled in Power BI, Python, SQL, ML, DL, NLP, Google Apps Script, Google Site

<p align="left">
  📢 <strong>Connect with me:</strong>&nbsp;

  <a href="https://www.linkedin.com/in/debasisbaidya">
    <img src="https://img.shields.io/badge/LinkedIn-View_Profile-blue?logo=linkedin&logoColor=white" />
  </a>

  <a href="mailto:speak2debasis@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-Mail_Me-red?logo=gmail&logoColor=white" />
  </a>

  <a href="https://api.whatsapp.com/send?phone=918013316086&text=Hi%20Debasis!">
    <img src="https://img.shields.io/badge/WhatsApp-Message-green?logo=whatsapp&logoColor=white" />
  </a>
</p>

---

## ⚙️ Try It Locally

```bash
# Run locally with image input
python main.py --image path/to/image.jpg

# Streamlit App
streamlit run Apps.py
```

---

## 💰 Cost & Resources

* 💻 Google Colab (Free Tier)
* 🚀 Training Time: \~2 hours for 200 epochs
* 💸 Total Cost: ₹0 (under free limits)

---

> 🚀 This project shows how machine learning, OCR, and a clean UI can turn static lab reports into structured, digital, and actionable data. Ideal for EMRs, diagnostic apps, and healthcare analytics. 🏥
