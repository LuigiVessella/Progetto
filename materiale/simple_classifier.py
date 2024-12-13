import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from collections import defaultdict
from PIL import Image

# Impostazioni
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "dataset/trainSetRGB")  # Path al dataset di addestramento
test_folder = os.path.join(base_dir, "dataset/testSetRGB")  # Path al dataset di test
batch_size = 1
num_epochs = 30
learning_rate = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trasformazioni delle immagini
transform = transforms.Compose([
    transforms.Resize((10, 10)),  # Ridimensiona le immagini a 10x10
    transforms.ToTensor(),        # Converti le immagini in tensori
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizzazione
])

# Caricamento del dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Suddivisione del dataset rispettando le classi
def split_dataset_by_class(dataset, split_ratio=0.8):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():
        split_point = int(len(indices) * split_ratio)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset

train_dataset, val_dataset = split_dataset_by_class(dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Modello CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 5 * 5, num_classes)  # Output da 10x10 ridotto a 5x5 dopo max pooling

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Creazione del modello
num_classes = len(dataset.classes)  # Numero di classi (label)
model = SimpleCNN(num_classes=num_classes).to(device)

# Ottimizzatore e funzione di perdita
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Funzione per addestramento
def train_model(model, train_loader, val_loader, num_epochs):
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

        # Validazione
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%")

# Addestramento del modello
train_model(model, train_loader, val_loader, num_epochs)

# Salvataggio del modello
torch.save(model.state_dict(), "gasf_classifier.pth")
print("Modello salvato come 'gasf_classifier.pth'.")

# Testing del modello su una cartella di immagini
def test_on_images_folder_with_subfolders_and_precision(model, folder_path, transform, class_names):
    model.eval()
    predictions = []
    correct = 0
    total = 0

    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        if not os.path.isdir(class_folder_path):
            continue

        image_paths = [os.path.join(class_folder_path, f) for f in os.listdir(class_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        with torch.no_grad():
            for image_path in image_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    image = transform(image).unsqueeze(0).to(device)
                    output = model(image)
                    _, predicted = torch.max(output, 1)
                    pred_class = class_names[predicted.item()]
                    image_name = os.path.basename(image_path)

                    # Controllo se il prefisso dell'immagine corrisponde alla classe predetta
                    image_prefix = image_name.split("_")[0]  # Ottieni il prefisso dal nome del file
                    if image_prefix == pred_class:
                        correct += 1

                    total += 1
                    predictions.append((image_name, pred_class))
                except Exception as e:
                    print(f"Errore con l'immagine {image_path}: {e}")

    # Calcolo e stampa della precisione
    precision = (correct / total) * 100 if total > 0 else 0
    print(f"Precisione basata sul prefisso del nome dell'immagine: {precision:.2f}%")
    return predictions, precision


# Caricamento del modello
try:
    model.load_state_dict(torch.load("gasf_classifier.pth"))
    model.to(device)
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")

# Predizioni
predictions, precision = test_on_images_folder_with_subfolders_and_precision(model, test_folder, transform, dataset.classes)
for image_name, pred_class in predictions:
    print(f"Immagine: {image_name}, Predizione: {pred_class}")
print(f"La precisione generale è : {precision}")
