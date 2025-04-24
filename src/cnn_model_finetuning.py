# =========================================================================================================
# Saideep Arikontham
# April 2025
# CS 5330 Final Project
# =========================================================================================================


import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna
import json

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

# CNN Model
class AccidentCNN(nn.Module):
    def __init__(self, conv_layers, linear_layers, dropout_rate, hidden_units):
        super().__init__()
        self.convs = nn.Sequential()
        in_channels = 3
        H, W = 224, 224

        for i in range(conv_layers):
            out_channels = 32 * (i + 1)
            self.convs.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.convs.add_module(f'relu{i}', nn.ReLU())
            self.convs.add_module(f'pool{i}', nn.MaxPool2d(2))
            H, W = H // 2, W // 2
            in_channels = out_channels

        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            x = self.convs(dummy)
            self.flat_dim = x.view(1, -1).size(1)

        self.dropout = nn.Dropout(dropout_rate)

        self.linear_layers = nn.Sequential()
        if linear_layers > 0:
            self.linear_layers.add_module("fc1", nn.Linear(self.flat_dim, hidden_units))
            self.linear_layers.add_module("relu1", nn.ReLU())
            for i in range(1, linear_layers):
                self.linear_layers.add_module(f"fc{i+1}", nn.Linear(hidden_units, hidden_units))
                self.linear_layers.add_module(f"relu{i+1}", nn.ReLU())
            final_in_features = hidden_units
        else:
            final_in_features = self.flat_dim

        self.output = nn.Linear(final_in_features, 2)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear_layers(x)
        return self.output(x)

# Objective function for Optuna
def objective(trial):
    conv_layers = trial.suggest_categorical("conv_layers", [1, 3, 5])
    linear_layers = trial.suggest_categorical("linear_layers", [1, 2, 4])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    hidden_units = trial.suggest_categorical("hidden_units", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 96])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 5, 20)

    device = "cpu"
    path = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/data/cctv_accident_data"
    model_dir = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/models/cnn_models"
    train_data, val_data = get_accident_data(path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    model = AccidentCNN(conv_layers, linear_layers, dropout_rate, hidden_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
            save_path = os.path.join(model_dir, f"cnn_accident_model_{trial.number}.pth")
            torch.save(best_model_state, save_path)

    model.load_state_dict(best_model_state)
    return val_accuracy

# Optuna controller and configuration saver
def run_optuna(n_trials):
    model_dir = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/models/cnn_models"
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

    best_trial = study.best_trial
    model_filename = f"cnn_accident_model_{best_trial.number}.pth"
    model_path = os.path.join(model_dir, model_filename)

    config = {
        "name": model_filename,
        "path": model_path,
        "conv_layers": best_trial.params["conv_layers"],
        "linear_layers": best_trial.params["linear_layers"],
        "dropout_rate": best_trial.params["dropout_rate"],
        "hidden_units": best_trial.params["hidden_units"]
    }

    config_path = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/config/cnn_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("\nBest trial:")
    print(f"  Accuracy: {best_trial.value:.4f}")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"\nConfiguration saved to: {config_path}")
    return study

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cnn_model_finetuning.py <num_trials>")
        sys.exit(1)

    try:
        n_trials = int(sys.argv[1])
    except ValueError:
        print("Please provide a valid integer for number of trials.")
        sys.exit(1)

    run_optuna(n_trials)
