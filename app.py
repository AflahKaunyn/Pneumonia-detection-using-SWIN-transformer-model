from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
import pandas as pd
import joblib
import os
import requests

app = Flask(__name__)

MODEL_PATH = "swin_pneumonia.pth"
MODEL_URL = "https://drive.google.com/uc?id=1uhR3BW7oKINHJuyDIVza-2Pha8Ug4_tG&export=download"

if not os.path.exists(MODEL_PATH):
    print("Downloading Swin Transformer model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded successfully.")

class SwinTransformerPneumonia(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformerPneumonia, self).__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


image_model = SwinTransformerPneumonia()
image_model.load_state_dict(torch.load("swin_pneumonia.pth", map_location=torch.device('cpu')))
image_model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])


LABELS = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]


csv_model = joblib.load("xgb_pneumonia_model.pkl")

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    prediction = None
    confidence = None
    csv_prediction = None
    csv_confidence = None

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

        # CSV
        if "csvfile" in request.files and request.files["csvfile"].filename != "":
            csv_file = request.files["csvfile"]
            try:
                df = pd.read_csv(csv_file)

                if df.shape[1] != 768:
                    csv_prediction = f"Error: CSV file must have exactly 768 columns. Uploaded file has {df.shape[1]} columns."
                    csv_confidence = None
                else:
                    features = df.iloc[:, :768].values
                    probabilities = csv_model.predict_proba(features)
                    confidence_score = probabilities.max(axis=1)[0]
                    predicted_class = probabilities.argmax(axis=1)[0]
                    csv_prediction = LABELS[predicted_class]
                    csv_confidence = f"{confidence_score * 100:.2f}%"
            except Exception as e:
                csv_prediction = "Error processing CSV file."
                csv_confidence = None
                print("CSV error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        csv_prediction=csv_prediction,
        csv_confidence=csv_confidence
    )

import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



