# =========================================================================================================
# Saideep Arikontham
# April 2025
# CS 5330 Final Project
# =========================================================================================================


import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
import json

# Config
VIDEO_PATH = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/live_test/accident_feed.mp4"
RESNET_CONFIG_PATH = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/config/resnet_config.json"
NUM_CLASSES = 2
INPUT_SIZE = 224
THRESHOLD = 0.5

# Load model
def load_model(config_path, device):
    with open(config_path, "r") as f:
        config = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(config["dropout_rate"]),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(config["path"], map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_frame(model, frame, device):
    img_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0][0]  # Class index 0 = Accident
    return probs

def draw_prediction(frame, prob, threshold):
    pred_label = "Accident" if prob >= threshold else "No Accident"
    label = f"Prediction: {pred_label} ({prob:.2f})"
    color = (0, 0, 255) if pred_label == "Accident" else (0, 255, 0)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)

def main():
    device = "cpu"
    model = load_model(RESNET_CONFIG_PATH, device)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/live_test/accident_labeled_output.mp4', fourcc, 20.0,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prob = predict_frame(model, rgb_frame, device)

        draw_prediction(frame, prob, threshold=THRESHOLD)

        cv2.imshow('Accident Detection', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved as 'accident_labeled_output.mp4'.")

if __name__ == "__main__":
    main()
