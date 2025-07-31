import gdown
import os

# Download model from Google Drive if not present
MODEL_PATH = "swin_pneumonia.pth"
DRIVE_URL = "https://drive.google.com/uc?id=1uhR3BW7oKINHJuyDIVza-2Pha8Ug4_tG"  # Direct download link

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm

# Set Streamlit page settings
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩺 Pneumonia Disease Prediction")
st.markdown("Upload a chest X-ray **image** to classify Pneumonia types.")

# Define class labels
LABELS = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]

# Define Swin Transformer model class
class SwinTransformerPneumonia(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformerPneumonia, self).__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Load model (cached)
@st.cache_resource
def load_model():
    model = SwinTransformerPneumonia()
    model.load_state_dict(torch.load("swin_pneumonia.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Image Upload
image_file = st.file_uploader("Upload Chest X-ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence_score, predicted_class = torch.max(probabilities, 1)
            prediction = LABELS[predicted_class.item()]
            confidence = f"{confidence_score.item() * 100:.2f}%"

        st.success(f"Prediction: **{prediction}**")
        st.info(f"Confidence: {confidence}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
