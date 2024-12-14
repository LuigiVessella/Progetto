import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os
from argparse import ArgumentParser

# Parsing degli argomenti dalla riga di comando
def parse_args():
    parser = ArgumentParser(description="Train and test a CNN model with Torchmetrics")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument('--train_data', type=str, default='dataset/trainSetRGB', help="Path to the training dataset")
    parser.add_argument('--test_data', type=str, default='dataset/testSetRGB', help="Path to the testing dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and validation")
    return parser.parse_args()

# Impostazioni
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trasformazioni delle immagini
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Caricamento del dataset
train_data = datasets.ImageFolder(root=args.train_data, transform=transform)
test_data = datasets.ImageFolder(root=args.test_data, transform=transform)

# Suddivisione dei dati in train e validation set
train_indices, val_indices = train_test_split(
    range(len(train_data)),
    test_size=0.2,
    stratify=[label for _, label in train_data]
)
train_dataset = Subset(train_data, train_indices)
val_dataset = Subset(train_data, val_indices)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Modello CNN
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Funzione di training
def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion):
    # Metriche
    train_accuracy = MulticlassAccuracy(num_classes=len(train_data.classes), average='macro').to(device)
    val_accuracy = MulticlassAccuracy(num_classes=len(train_data.classes), average='macro').to(device)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        train_acc = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_acc += train_accuracy(outputs, labels)

        epoch_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += val_accuracy(outputs, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={epoch_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# Funzione di testing
def test_model(model, test_loader):
    precision = MulticlassPrecision(num_classes=len(test_data.classes), average=None).to(device)
    recall = MulticlassRecall(num_classes=len(test_data.classes), average=None).to(device)
    f1_score = MulticlassF1Score(num_classes=len(test_data.classes), average=None).to(device)

    total_samples = 0
    all_precisions = torch.zeros(len(test_data.classes)).to(device)
    all_recalls = torch.zeros(len(test_data.classes)).to(device)
    all_f1s = torch.zeros(len(test_data.classes)).to(device)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Accumula le metriche normalizzando per ogni batch
            all_precisions += precision(preds, labels)
            all_recalls += recall(preds, labels)
            all_f1s += f1_score(preds, labels)
            total_samples += 1

    # Media delle metriche
    all_precisions /= total_samples
    all_recalls /= total_samples
    all_f1s /= total_samples

    print("Class-wise Metrics:")
    for i, class_name in enumerate(test_data.classes):
        print(f"{class_name}: Precision={all_precisions[i] * 100:.2f}%, "
              f"Recall={all_recalls[i] * 100:.2f}%, "
              f"F1-Score={all_f1s[i] * 100:.2f}%")

# Setup modello, ottimizzatore e funzione di perdita
model = ImprovedCNN(num_classes=len(train_data.classes)).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

# Training
train_model(model, train_loader, val_loader, args.num_epochs, optimizer, criterion)

# Testing
test_model(model, test_loader)
