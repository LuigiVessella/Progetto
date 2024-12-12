import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Impostazioni
data_dir = "materiale/datiOriginali_GASF"  # Path al dataset
batch_size = 32
num_epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trasformazioni delle immagini
transform = transforms.Compose([
    transforms.Resize((10, 10)),  # Ridimensiona le immagini a 10x10 (se necessario)
    transforms.ToTensor(),       # Converti le immagini in tensori
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizzazione
])

# Caricamento del dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Divisione del dataset in train e validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
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

#per testarlo:
'''
# Carica i pesi del modello
model = SimpleCNN(num_classes=num_classes)  # Inizializza il modello con la stessa architettura
model.load_state_dict(torch.load("gasf_classifier.pth"))
model.eval()  # Imposta il modello in modalità di valutazione

'''


#su dataset test:
'''
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carica il dataset di test
test_dataset = datasets.ImageFolder(root="path_to_test_dataset", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Valutazione sul dataset di test
all_labels = []
all_predictions = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Salva le predizioni e le etichette reali per ulteriori analisi
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuratezza sul dataset di test: {accuracy:.2f}%")

# Report di classificazione
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=test_dataset.classes))

# Matrice di confusione
conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



'''
