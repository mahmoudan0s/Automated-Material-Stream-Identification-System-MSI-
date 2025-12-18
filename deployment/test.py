import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path
import joblib

UNKNOWN_LABEL = 6
CONF_THRESHOLD = 0.55
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    0: 0,  # Glass
    1: 1,  # Paper
    2: 2,  # Cardboard
    3: 3,  # Plastic
    4: 4,  # Metal
    5: 5   # Trash
}

def predict_with_unknown(X, svm, threshold=0.55, unknown_label=6):
    probs = svm.predict_proba(X)            # shape (N, num_classes)
    max_probs = probs.max(axis=1)
    preds = svm.classes_[probs.argmax(axis=1)]

    final_preds = []
    for p, conf in zip(preds, max_probs):
        if conf < threshold:
            final_preds.append(unknown_label)
        else:
            final_preds.append(int(p))

    return np.array(final_preds), max_probs

def predict(dataFilePath, bestModelPath):
    # -------- Load models --------
    svm = joblib.load(bestModelPath)
    pca = joblib.load(bestModelPath.replace("svm_model.pkl", "pca_model.pkl"))

    # Feature extractor
    model = models.efficientnet_b0(weights="DEFAULT")
    model.classifier = nn.Identity()  # remove classifier head
    model = model.to(DEVICE)
    model.eval()

    # -------- Preprocessing --------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -------- Load and extract features --------
    image_files = sorted([
        f for f in os.listdir(dataFilePath)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    feats = []
    for img_name in image_files:
        img_path = os.path.join(dataFilePath, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy()
            feats.append(feat[0])

    feats = np.array(feats)
    feats_pca = pca.transform(feats)

    # -------- Open-set prediction --------
    predictions, _ = predict_with_unknown(feats_pca, svm, threshold=CONF_THRESHOLD, unknown_label=UNKNOWN_LABEL)

    return predictions.tolist()


if __name__ == "__main__":
    # Example usage
    test_folder = r"C:\Users\DELL\myGithub\Automated-Material-Stream-Identification-System-MSI-\deployment\testing" # replace with your test folder
    svm_model_path = r"C:\Users\DELL\myGithub\Automated-Material-Stream-Identification-System-MSI-\models\svm_model.pkl"# replace if path is different

    preds = predict(test_folder, svm_model_path)
    print("Predictions:", preds)
