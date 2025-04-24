# =========================================================================================================
# Saideep Arikontham
# April 2025
# CS 5330 Final Project
# =========================================================================================================


import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import optuna

# Dataset setup
def get_accident_data(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(os.path.join(path, 'train'), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(path, 'val'), transform=transform)

    # Ensure Accident is 1, Non Accident is 0
    class_to_idx = {'Non Accident': 0, 'Accident': 1}
    train_data.class_to_idx = class_to_idx
    val_data.class_to_idx = class_to_idx
    
    return train_data, val_data

# Build ResNet18 model with custom head
def build_resnet18(dropout_rate, freeze_layers, device):
    model = models.resnet18(pretrained=True)

    # Freeze layers
    layer_count = 0
    for child in model.children():
        if layer_count < freeze_layers:
            for param in child.parameters():
                param.requires_grad = False
        layer_count += 1

    # Replace final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_ftrs, 2)
    )

    return model.to(device)

# Optuna objective function
def objective(trial):
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    freeze_layers = trial.suggest_int("freeze_layers", 0, 5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)

    device = "cpu"
    path = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/data/cctv_accident_data"
    model_dir = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/models/resnet_models"
    train_data, val_data = get_accident_data(path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    model = build_resnet18(dropout_rate, freeze_layers, device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss, total_train_samples = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X.size(0)
            total_train_samples += X.size(0)

        avg_train_loss = total_train_loss / total_train_samples

        model.eval()
        total_val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                total_val_loss += loss.item() * X.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val_loss = total_val_loss / total
        val_accuracy = correct / total

        print(f"Trial {trial.number} | Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            save_path = os.path.join(model_dir, f"resnet_accident_model_{trial.number}.pth")
            torch.save(best_model_state, save_path)

    model.load_state_dict(best_model_state)
    return val_accuracy

# Run Optuna
def run_optuna(n_trials):
    model_dir = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/models/resnet_models"
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            file_path = os.path.join(model_dir, filename)
            if os.path.isfile(file_path) and filename.endswith(".pth"):
                os.remove(file_path)
        print(f"Cleared previous models in {model_dir}")
    else:
        os.makedirs(model_dir)
        print(f"Created model directory at {model_dir}")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\nBest trial:")
    print(f"  Accuracy: {study.best_value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save configuration of best model
    best_trial = study.best_trial
    best_model_name = f"resnet_accident_model_{best_trial.number}.pth"
    best_model_path = os.path.join(model_dir, best_model_name)

    config = {
        "name": best_model_name,
        "path": best_model_path,
        "dropout_rate": best_trial.params["dropout_rate"]
    }

    config_path = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/config/resnet_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nConfiguration saved to: {config_path}")
    return study

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resnet18_finetuning.py <num_trials>")
        sys.exit(1)

    try:
        n_trials = int(sys.argv[1])
    except ValueError:
        print("Please provide a valid integer for number of trials.")
        sys.exit(1)

    run_optuna(n_trials)
