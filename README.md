# Multimodal-AI-for-Chest-X-ray-Diagnosis-and-Report-Synthesis

This project presents a deep learning pipeline combining image classification (CNN), synthetic image generation (GAN), and report synthesis (T5 transformer) for automated diagnosis of pneumonia from chest X-ray images.

The goal is to build a complete AI system that:
- Classifies X-rays as `Pneumonia` or `Normal`
- Generates synthetic `Normal` chest X-rays using GAN
- Produces a professional radiology report using a pre-trained T5 model

---

## Key Features

- **CNN-based classifier**: Detects pneumonia from chest X-rays.
- **GAN-based generator**: Learns to generate synthetic normal X-ray images.
- **T5-based report generator**: Creates structured radiology reports with findings, impressions, and recommendations.
- **Interactive Gradio UI**: Upload images and get real-time predictions and auto-generated reports.

---

## Dataset

- **Source**: [Kaggle - Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: Normal, Pneumonia
- **Total**: ~5,000 chest X-ray images
- **Structure**: `train/`, `val/`, and `test/` folders

---

## Technologies Used
Python 3.11
TensorFlow / Keras
HuggingFace Transformers
Gradio
OpenCV
Google Colab + Drive
GANs (DCGAN-style architecture)


