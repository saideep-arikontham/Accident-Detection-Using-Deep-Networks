# =========================================================================================================
# Saideep Arikontham
# April 2025
# CS 5330 Final Project
# =========================================================================================================

import sys
import os
import random
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score


# =========================================================================================================
# CNN Class Definition
# =========================================================================================================

class FlexibleCNN(nn.Module):
    """
    A flexible CNN architecture for accident detection from CCTV images.
    """

    def __init__(self, conv_layers, linear_layers, dropout_rate, hidden_units):
        super().__init__()
        self.activation = F.relu
        self.convs = nn.Sequential()
        in_channels = 3  # For RGB images
        H, W = 224, 224

        for i in range(conv_layers):
            out_channels = 32 * (i + 1)
            self.convs.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.convs.add_module(f'relu{i}', nn.ReLU())
            if H >= 2 and W >= 2:
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

        self.final_layer = nn.Linear(final_in_features, 2)  # 2 classes: Accident, Non-Accident

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear_layers(x)
        return self.final_layer(x)


# =========================================================================================================
# Data Loading
# =========================================================================================================

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


# =========================================================================================================
# Parameter Grid Setup
# =========================================================================================================

def get_param_grid(max_conv_layers, max_lin_layers, max_dropout, max_hidden_units, max_batch_size, learning_rate, max_epochs):
    conv_layers_list = list(range(1, max_conv_layers + 1))
    linear_layers_list = list(range(1, max_lin_layers + 1))
    dropouts = np.arange(0.1, max_dropout + 0.1, 0.1).tolist()
    hidden_units_list = list(range(64, max_hidden_units + 1, 64))
    batch_sizes = list(range(32, max_batch_size + 1, 32))
    epochs_list = list(range(5, max_epochs + 1, 5))

    print("Parameter grid:")
    print("Conv layers:", conv_layers_list)
    print("Linear layers:", linear_layers_list)
    print("Dropouts:", dropouts)
    print("Hidden units:", hidden_units_list)
    print("Batch sizes:", batch_sizes)
    print("Learning rates:", learning_rate)
    print("Epochs:", epochs_list)

    grid = list(product(conv_layers_list, linear_layers_list, dropouts, hidden_units_list, batch_sizes, learning_rate, epochs_list))
    print("Total combinations:", len(grid))

    random.shuffle(grid)
    return grid[:50]


# =========================================================================================================
# Train and Evaluate
# =========================================================================================================

def train_and_evaluate(config, path):
    device = "cpu"

    conv_layers, linear_layers, dropout_rate, hidden_units, batch_size, learning_rate, epochs = config
    train_data, val_data = get_accident_data(path)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    model = FlexibleCNN(conv_layers, linear_layers, dropout_rate, hidden_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    duration = time.time() - start_time

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    return {
        'conv_layers': conv_layers,
        'linear_layers': linear_layers,
        'dropout_rate': dropout_rate,
        'hidden_units': hidden_units,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'val_acc': acc,
        'time_sec': round(duration, 2)
    }


# =========================================================================================================
# Grid Execution
# =========================================================================================================

def run_train_loop_for_all_configs(path, grid):
    results = []
    for config in grid:
        print("Running config:", config)
        result = train_and_evaluate(config, path)
        print("Validation Accuracy: {:.2f}% | Time: {}s".format(result['val_acc'], result['time_sec']))
        results.append(result)
    return pd.DataFrame(results).sort_values(by='val_acc', ascending=False)


# =========================================================================================================
# Main Function
# =========================================================================================================

def main(argv):
    if len(argv) < 7:
        print("Usage: python accident_network.py <max_conv_layers> <max_lin_layers> <max_dropout> <max_hidden_units> <max_batch_size> <max_epochs>")
        return

    max_conv_layers = int(argv[1])
    max_lin_layers = int(argv[2])
    max_dropout = float(argv[3])
    max_hidden_units = int(argv[4])
    max_batch_size = int(argv[5])
    max_epochs = int(argv[6])
    learning_rate = [0.01, 0.001, 0.0001]

    path = '/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/data/cctv_accident_data'

    # Just ensure dataset is downloaded/loaded
    get_accident_data(path)

    grid = get_param_grid(max_conv_layers, max_lin_layers, max_dropout, max_hidden_units, max_batch_size, learning_rate, max_epochs)
    result_df = run_train_loop_for_all_configs(path, grid)
    print(result_df)
    return

if __name__ == "__main__":
    main(sys.argv)
