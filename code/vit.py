import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import timm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load EuroSAT dataset
dataset = EuroSAT(root='./data', transform=transform, download=True)
class_names = dataset.classes
print(f"Classes: {class_names}")

# Train-validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Sobel Edge Detection
class SobelEdgeDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3).to(device)

    def forward(self, x):
        x_gray = torch.mean(x, dim=1, keepdim=True) if x.shape[1] > 1 else x
        edge_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        edges = edges / (edges.amax(dim=[2, 3], keepdim=True) + 1e-8)
        return edges.repeat(1, x.shape[1], 1, 1)

# Edge-aware CNN
class EdgeAwareCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.edge_detection = SobelEdgeDetection()
        self.edge_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.edge_output = nn.Conv2d(16, in_channels, kernel_size=1)

    def forward(self, x):
        edges = self.edge_detection(x)
        edge_features = self.edge_cnn(edges)
        edge_output = self.edge_output(edge_features)
        return x + edge_output, edges, edge_features

# Vision Transformer with edge-enhancement
class EdgeEnhancedViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.edge_enhancer = EdgeAwareCNN()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x, _, _ = self.edge_enhancer(x)
        x = F.interpolate(x, size=(224, 224))
        return self.vit(x)

# Initialize model
model = EdgeEnhancedViT(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    return acc, f1, prec, rec, y_true, y_pred

# Training loop
def train_model(model, train_loader, val_loader, epochs=25, checkpoint_path='best_model.pth'):
    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        scheduler.step()
        acc, f1, prec, rec, y_true, y_pred = evaluate(model, val_loader)
        print(f"\nValidation -> Acc: {acc:.2f}% | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✔️ Model saved at epoch {epoch+1} with best F1: {f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.show()

# Start training
train_model(model, train_loader, val_loader, epochs=18)