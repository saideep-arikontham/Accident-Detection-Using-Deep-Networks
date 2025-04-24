# =========================================================================================================
# Saideep Arikontham
# April 2025
# CS 5330 Final Project
# =========================================================================================================


import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import numpy as np

# Configuration
RESNET_CONFIG_PATH = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/config/resnet_config.json"
DATA_DIR = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/data/cctv_accident_data"
BATCH_SIZE = 64
INPUT_SIZE = 224
NUM_CLASSES = 2

def get_data_loader(data_dir, batch_size, input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    # Update to ensure 'Accident' is 0 and 'Non Accident' is 1
    class_to_idx = {'Accident': 0, 'Non Accident': 1}
    val_ds.class_to_idx = class_to_idx

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return val_loader

def evaluate_at_thresholds(model, loader, device, step=0.05):
    model.eval()
    y_true, y_probs = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Now we extract the probability for class index 0 (Accident)
            probs = torch.softmax(outputs, dim=1)[:, 0]
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    thresholds = np.arange(0.0, 1.05, step)
    best_f1 = 0
    best_thresh = 0.5

    print("\nThreshold-wise Evaluation (Validation Set):")
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 55)

    for thresh in thresholds:
        y_pred = (np.array(y_probs) >= thresh).astype(int)
        # Since 1 = Accident in `y_pred` but Accident is 0 in `y_true`, we need to invert predictions
        y_pred = 1 - y_pred
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

        print(f"{thresh:<10.2f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")

    print(f"\nBest Threshold based on F1 Score: {best_thresh:.2f} (F1 = {best_f1:.4f})")
    return best_thresh

def main():
    device = "cpu"

    with open(RESNET_CONFIG_PATH, "r") as f:
        config = json.load(f)

    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Sequential(
        nn.Dropout(config["dropout_rate"]),
        nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    )
    resnet.load_state_dict(torch.load(config["path"], map_location=device))
    resnet.to(device)

    val_loader = get_data_loader(DATA_DIR, BATCH_SIZE, INPUT_SIZE)

    best_thresh = evaluate_at_thresholds(resnet, val_loader, device)

if __name__ == "__main__":
    main()
