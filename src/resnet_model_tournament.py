# =========================================================================================================
# Saideep Arikontham
# April 2025
# CS 5330 Final Project
# =========================================================================================================


import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd


def get_data_loaders(train_dir, val_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # Ensure Accident is 1, Non Accident is 0
    class_to_idx = {'Non Accident': 0, 'Accident': 1}
    train_dataset.class_to_idx = class_to_idx
    val_dataset.class_to_idx = class_to_idx

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def train_and_evaluate(model, train_loader, val_loader, device, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")

        history.append({
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Val Loss": avg_val_loss,
            "Val Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    return history


def main():
    # Get command line arguments
    if len(sys.argv) != 3:
        print("Usage: python resnet_model_tournament.py <num_epochs> <learning_rate>")
        return

    num_epochs = int(sys.argv[1])
    learning_rate = float(sys.argv[2])


    # Set data directory
    data_dir = '/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/data/cctv_accident_data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Set device
    device = "cpu"

    # Load data
    train_loader, val_loader = get_data_loaders(train_dir, val_dir)

    # Model variants
    resnet_variants = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50
    }

    results = []

    for name, model_fn in resnet_variants.items():
        print(f"\nTraining {name.upper()}...\n")
        model = model_fn(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)

        history = train_and_evaluate(model, train_loader, val_loader, device, num_epochs, learning_rate)

        best_epoch = min(history, key=lambda x: x['Val Loss'])
        results.append({
            "Model": name,
            **best_epoch
        })

    # Results summary
    results_df = pd.DataFrame(results)
    print("\nFinal Results Summary:")
    print(results_df.sort_values(by='Val Accuracy', ascending=False).to_string(index=False))

    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["Val Accuracy"])
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
