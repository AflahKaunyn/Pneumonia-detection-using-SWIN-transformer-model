import gdown
import os

MODEL_PATH = "swin_pneumonia.pth"
DRIVE_URL = "https://drive.google.com/file/d/1uhR3BW7oKINHJuyDIVza-2Pha8Ug4_tG/view?usp=drive_link"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)


import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
import pandas as pd
import joblib
import numpy as np

# Title
st.set_page_config(page_title="Pneumonia Detection App", layout="centered")
st.title("🩺 Pneumonia Disease Prediction")
st.markdown("Upload a chest X-ray **image** or a **CSV file** of features to classify Pneumonia types.")

# Define the model class
class SwinTransformerPneumonia(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformerPneumonia, self).__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Load models
@st.cache_resource
def load_models():
    image_model = SwinTransformerPneumonia()
    image_model.load_state_dict(torch.load("swin_pneumonia.pth", map_location=torch.device("cpu")))
    image_model.eval()
    csv_model = joblib.load("xgb_pneumonia_model.pkl")
    return image_model, csv_model

image_model, csv_model = load_models()

# Label names
LABELS = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Image Upload
st.header("🔍 Image-based Prediction")
image_file = st.file_uploader("Upload Chest X-ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = image_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence_score, predicted_class = torch.max(probabilities, 1)
            prediction = LABELS[predicted_class.item()]
            confidence = f"{confidence_score.item() * 100:.2f}%"

        st.success(f"Prediction: **{prediction}**")
        st.info(f"Confidence: {confidence}")

    except Exception as e:
        st.error(f"Image Processing Error: {e}")

# CSV Upload
st.header("📊 CSV-based Prediction")
csv_file = st.file_uploader("Upload CSV File (768 features)", type=["csv"])

if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)

        if df.shape[1] != 768:
            st.error(f"Expected 768 columns, but got {df.shape[1]}")
        else:
            features = df.iloc[:, :768].values
            probabilities = csv_model.predict_proba(features)
            confidence_score = probabilities.max(axis=1)[0]
            predicted_class = probabilities.argmax(axis=1)[0]

            csv_prediction = LABELS[predicted_class]
            csv_confidence = f"{confidence_score * 100:.2f}%"

            st.success(f"CSV Prediction: **{csv_prediction}**")
            st.info(f"Confidence: {csv_confidence}")

    except Exception as e:
        st.error(f"CSV Processing Error: {e}")
