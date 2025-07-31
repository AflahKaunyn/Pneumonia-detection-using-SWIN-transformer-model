# 🧠 Pneumonia Detection Using Swin Transformer

This project uses a Swin Transformer-based deep learning model to classify chest X-ray images into three categories:
- Normal
- Bacterial Pneumonia
- Viral Pneumonia

It is designed to assist in automated diagnosis of pneumonia using medical imaging.

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
pip install -r requirements.txt

---

# Model Training
The model is based on torchvision.models.swin_t.

Final classification layer is modified for 3 classes.

Trained using a labeled chest X-ray dataset divided into train/, val/, and test/ directories.

---

 📜 License

This project is intended for **educational and research purposes only**.  
Feel free to use or modify the code for non-commercial use with proper credit.  
No warranties provided.
