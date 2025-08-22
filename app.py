from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import requests

app = Flask(__name__)

# Download the Swin model file from Google Drive if not present
SWIN_MODEL_URL = "https://drive.google.com/uc?id=1uhR3BW7oKINHJuyDIVza-2Pha8Ug4_tG&export=download"
SWIN_MODEL_FILE = "swin_pneumonia.pth"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"{filename} downloaded.")

download_file(SWIN_MODEL_URL, SWIN_MODEL_FILE)

# Define the Swin Transformer Model class
class SwinTransformerPneumonia(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformerPneumonia, self).__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# Initialize the model
image_model = SwinTransformerPneumonia()
image_model.load_state_dict(torch.load(SWIN_MODEL_FILE, map_location=torch.device('cpu')))
image_model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LABELS = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    prediction = None
    confidence = None

    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            try:
                image = Image.open(file).convert("RGB")
                image = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = image_model(image)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence_score, predicted_class = torch.max(probabilities, 1)
                    prediction = LABELS[predicted_class.item()]
                    confidence = f"{confidence_score.item() * 100:.2f}%"
            except Exception as e:
                prediction = "Error processing image."
                confidence = None
                print("Image error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
