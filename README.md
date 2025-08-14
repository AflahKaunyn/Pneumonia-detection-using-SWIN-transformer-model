# 🩺  Pneumonia Detection Using Swin Transformer


This project uses a Swin Transformer-based deep learning model to classify chest X-ray images into three categories:
- Normal
- Bacterial Pneumonia
- Viral Pneumonia
 
- 📷 Chest X-ray images via Swin Transformer
- 📊 Clinical CSV data via XGBoost classifier
  
# 🩺  Download model using the link provide 
"https://drive.google.com/uc?id=1uhR3BW7oKINHJuyDIVza-2Pha8Ug4_tG&export=download"

---

## ⚙️ How It Works

1. A chest X-ray image is preprocessed: resized and normalized.
2. The processed image is passed to a trained Swin Transformer model.
3. The model predicts the class of pneumonia (or normal).

---

## 🔧 Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/AflahKaunyn/Pneumonia-detection-using-SWIN-transformer-model.git
cd Pneumonia-detection-using-SWIN-transformer-model

## Run locally
pip install -r requirements.txt
streamlit run app.py


---

### 🔗 Model Weights

⚠️ The Swin Transformer model file is not included in this repository due to size limitations.

The file will be **automatically downloaded** from Google Drive when you run `app.py`.

If needed, you can manually download it from:
[Download from Google Drive (if needed)](https://drive.google.com/file/d/1uhR3BW7oKINHJuyDIVza-2Pha8Ug4_tG/view?usp=drive_link)

---

# Model Training
The model is based on torchvision.models.swin_t.

Final classification layer is modified for 3 classes.

Trained using a labeled chest X-ray dataset divided into train/, val/, and test/ directories.

---

 📜 License

This project is intended for educational and research purposes only.  
Feel free to use or modify the code for non-commercial use with proper credit.  
No warranties provided.
