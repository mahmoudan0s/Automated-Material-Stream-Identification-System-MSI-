
# -*- coding: utf-8 -*-
"""
Live camera classification (EfficientNet-B0 features -> Scaler -> PCA -> SVM)
- Unknown rejection by probability threshold
- Consistent class ID mapping
- Lightweight preprocessing for better FPS
"""

import os
import time
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# ---------- Device ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Paths (adjust to your environment) ----------
SVM_PATH   = r"C:\Users\DELL\myGithub\Automated-Material-Stream-Identification-System-MSI-\models\svm_model.pkl"
PCA_PATH   = r"C:\Users\DELL\myGithub\Automated-Material-Stream-Identification-System-MSI-\models\pca_model.pkl"
SCALER_PATH= None  # if you saved one
# If you did not save a scaler, set SCALER_PATH=None and skip scaling in code.

# ---------- Class mapping (fixed order) ----------
ID_TO_NAME = {
    0: "Glass",
    1: "Paper",
    2: "Cardboard",
    3: "Plastic",
    4: "Metal",
    5: "Trash",
    6: "Unknown",
}


UNKNOWN_LABEL  = 6
CONF_THRESHOLD = 0.75  # raise if you want stricter Unknown rejection

# ---------- EfficientNet input ----------
INPUT_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def load_models():
    # EfficientNet-B0 backbone for features
    eff = models.efficientnet_b0(weights="DEFAULT")
    eff.classifier = nn.Identity()  # remove final classifier
    eff.to(DEVICE).eval()

    # SVM / PCA / scaler (trained offline)
    svm = joblib.load(SVM_PATH)
    pca = joblib.load(PCA_PATH)
    scaler = None
    if SCALER_PATH and os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    return eff, svm, pca, scaler

@torch.no_grad()
def extract_feat(eff_model, img_tensor):
    """Return a (1280,) EfficientNet-B0 feature vector."""
    img_tensor = img_tensor.to(DEVICE)
    feats = eff_model.features(img_tensor)                          # (1, C, H', W')
    feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))  # (1, C, 1, 1)
    feats = feats.flatten(1)                                        # (1, C)
    return feats.cpu().numpy()[0]                                   # (C,)

def preprocess(frame_bgr):
    """Convert frame to tensor ready for EfficientNet."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return transform(pil_img).unsqueeze(0)  # (1, 3, 224, 224)

def postprocess(feat_vec, scaler, pca):
    """Apply the same scaler/PCA used in training."""
    X = feat_vec.reshape(1, -1)   # (1, 1280)
    if scaler is not None:
        X = scaler.transform(X)
    Xp = pca.transform(X)         # (1, K)
    return Xp

def predict_id_only(svm, Xp, threshold=CONF_THRESHOLD, unknown_label=UNKNOWN_LABEL):
    """
    Return ONLY the class ID (int). Uses probability threshold for Unknown.
    """
    probs = svm.predict_proba(Xp)[0]             # (C,)
    maxp  = probs.max()
    pred  = int(svm.classes_[probs.argmax()])    # IDs from training label space
    if maxp < threshold:
        return unknown_label
    return pred

def main():
    eff, svm, pca, scaler = load_models()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    print("🎥 Live camera started. Press 'q' to exit.")
    prev_t, fps = time.time(), 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor = preprocess(frame)
        feat_vec   = extract_feat(eff, img_tensor)
        Xp         = postprocess(feat_vec, scaler, pca)
        pred_id    = predict_id_only(svm, Xp, threshold=CONF_THRESHOLD, unknown_label=UNKNOWN_LABEL)

        # Overlay ONLY the integer class ID
        cv2.putText(frame, f"{pred_id}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # (Optional) show FPS small in corner; comment these two lines if you want strictly only the number
        curr_t = time.time()
        dt = curr_t - prev_t
        if dt > 0: fps = 1.0 / dt
        prev_t = curr_t
        cv2.putText(frame, f"FPS:{fps:.1f}", (frame.shape[1]-120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Live Camera", frame)
        if (cv2.waitKey(1) & 0xFF) == 'q':
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Camera closed.")

if __name__ == "__main__":
    main()
