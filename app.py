from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import requests

app = Flask(__name__)

# Google Drive download helpers for large files
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Swin model file and your Google Drive file ID extracted from your link
SWIN_MODEL_FILE = "swin_pneumonia.pth"
SWIN_MODEL_ID = "1uhR3BW7oKINHJuyDIVza-2Pha8Ug4_tG"

# Download model if not already present
if not os.path.exists(SWIN_MODEL_FILE):
    print(f"Downloading {SWIN_MODEL_FILE} from Google Drive...")
    download_file_from_google_drive(SWIN_MODEL_ID, SWIN_MODEL_FILE)
    print(f"Downloaded {SWIN_MODEL_FILE}.")

# Define Swin Transformer pneumonia model class
class SwinTransformerPneumonia(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformerPneumonia, self).__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# Load the model weights
image_model = SwinTransformerPneumonia()
image_model.load_state_dict(torch.load(SWIN_MODEL_FILE, map_location=torch.device('cpu'), weights_only=False))
image_model.eval()

# Image preprocessing transformations
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
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)


