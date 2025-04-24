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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve

# Paths
RESNET_MODEL_PATH = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/models/resnet_models/resnet_accident_model_1.pth"
CNN_MODEL_PATH = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/models/cnn_models/cnn_accident_model_37.pth"
DATA_DIR = "/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/data/cctv_accident_data"
BATCH_SIZE = 64
NUM_CLASSES = 2
INPUT_SIZE = 224

# CNN model with exact architecture used during training
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

def get_data_loaders(data_dir, batch_size, input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    # Ensure Accident is 1, Non Accident is 0
    class_to_idx = {'Non Accident': 0, 'Accident': 1}
    val_ds.class_to_idx = class_to_idx
    test_ds.class_to_idx = class_to_idx
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return val_loader, test_loader

def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probs)
    }
    return metrics, y_true, y_pred, y_probs



# New function to plot confusion matrix and ROC curve individually for each model and dataset
def plot_confusion_and_roc_individual(y_true_pred_probs_dict, dataset_name):
    for name, (y_true, y_pred, y_probs) in y_true_pred_probs_dict.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[0])
        axes[0].set_title(f"{dataset_name} Confusion Matrix - {name}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_probs):.2f}")
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_title(f"{dataset_name} ROC Curve - {name}")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


def main():
    device = "cpu"
    val_loader, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE, INPUT_SIZE)

    # Load separate configs
    with open("/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/config/cnn_config.json", "r") as f:
        cnn_config = json.load(f)
    with open("/Users/saideepbunny/Coursework/PRCV/Accident-Detection-Using-Deep-Networks/config/resnet_config.json", "r") as f:
        resnet_config = json.load(f)

    # Build and load ResNet model
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Sequential(
        nn.Dropout(resnet_config["dropout_rate"]),
        nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    )
    resnet.load_state_dict(torch.load(resnet_config["path"], map_location=device))
    resnet.to(device)

    # Build and load CNN model
    cnn = AccidentCNN(
        conv_layers=cnn_config["conv_layers"],
        linear_layers=cnn_config["linear_layers"],
        dropout_rate=cnn_config["dropout_rate"],
        hidden_units=cnn_config["hidden_units"]
    )
    cnn.load_state_dict(torch.load(cnn_config["path"], map_location=device))
    cnn.to(device)

    # Evaluate on validation set
    print("Validation Set Metrics:")
    val_results = {}
    for name, model in [("ResNet18", resnet), ("CNN", cnn)]:
        metrics, y_true, y_pred, y_probs = evaluate_model(model, val_loader, device)
        val_results[name] = (y_true, y_pred, y_probs)
        print(f"{name}: {metrics}")
    plot_confusion_and_roc_individual(val_results, dataset_name="Validation")

    # Evaluate on test set
    print("\nTest Set Metrics:")
    test_results = {}
    for name, model in [("ResNet18", resnet), ("CNN", cnn)]:
        metrics, y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
        test_results[name] = (y_true, y_pred, y_probs)
        print(f"{name}: {metrics}")
    plot_confusion_and_roc_individual(test_results, dataset_name="Test")


if __name__ == "__main__":
    main()
