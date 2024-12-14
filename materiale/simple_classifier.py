#comando per eseguire lo script:
# python3 materiale/simple_classifier.py --num_epochs 50 --learning_rate 0.0001 --train_data "dataset/trainSetRGB" --test_data "dataset/testSetRGB" --batch_sizes 1 4 8 16

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from collections import defaultdict
from PIL import Image
import argparse

# Parsing degli argomenti dalla riga di comando
def parse_args():
    parser = argparse.ArgumentParser(description="Train and test a CNN model")
    
    # Aggiungi i parametri di configurazione
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument('--train_data', type=str, default='dataset/trainSetRGB', help="Path to the training dataset")
    parser.add_argument('--test_data', type=str, default='dataset/testSetRGB', help="Path to the testing dataset")
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16], help="List of batch sizes to use for training")

    return parser.parse_args()

# Impostazioni
args = parse_args()

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, args.train_data)  # Path al dataset di addestramento (passato come parametro)
test_folder = os.path.join(base_dir, args.test_data)  # Path al dataset di test (passato come parametro)
num_epochs = args.num_epochs
learning_rate = args.learning_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trasformazioni delle immagini
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Aumentato a 64x64 per ottenere più informazioni
    transforms.ToTensor(),        # Converti le immagini in tensori
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizzazione per RGB
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

# Modello CNN migliorato
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        
        # Primo blocco convoluzionale
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Secondo blocco convoluzionale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Terzo blocco convoluzionale
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Ridimensionato per immagini 64x64
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout per prevenire overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Conv + BN + ReLU
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))  # Fully connected
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)  # Output
        return x

# Funzione per addestramento
# Funzione per addestramento con early stopping
def train_model(model, train_loader, val_loader, num_epochs, file, early_stop_patience=10):
    best_val_accuracy = 0  # Per memorizzare la migliore perdita di validazione
    patience_counter = 0  # Contatore per early stopping
    
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

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        val_accuracy = 100 * correct / total

        file.write(f"Epoch {epoch+1}/{num_epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Accuracy: {val_accuracy:.2f}%\n")

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0  # Resetta il contatore
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                file.write(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy.\n")
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break


# Funzione di testing
def test_on_images_folder_with_subfolders_and_precision(model, folder_path, transform, class_names, file):
    model.eval()
    predictions = []
    correct = 0
    total = 0

    # Inizializzazione di un dizionario per contare i corretti per ogni classe
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

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
                        class_correct[pred_class] += 1  # Aggiungi al conteggio corretti per la classe
                    class_total[pred_class] += 1  # Aggiungi al conteggio totale per la classe

                    total += 1
                    predictions.append((image_name, pred_class))
                except Exception as e:
                    print(f"Errore con l'immagine {image_path}: {e}")

    # Calcolo della precisione per ogni classe
    precision = (correct / total) * 100 if total > 0 else 0
    file.write(f"\nPrecisione generale per batch_size={batch_size}: {precision:.2f}%\n")
    for class_name in class_names:
        if class_total[class_name] > 0:
            class_precision = (class_correct[class_name] / class_total[class_name]) * 100
            file.write(f"{class_name}: {class_precision:.2f}%\n")
        else:
            file.write(f"{class_name}: 0.00% (nessun esempio nella classe)\n")

    return predictions, precision, class_total, class_correct

# Ciclo sui vari batch_size
with open("training_results.txt", "w") as file:  # Apre il file per scrivere
    for batch_size in args.batch_sizes:  # Ora itera sui batch_sizes passati come parametro
        file.write(f"\n### Risultati per batch_size={batch_size} ###\n")
        print(f"\nInizio addestramento con batch_size={batch_size}")
        
        # Creazione del DataLoader con il batch_size corrente
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Creazione del modello
        model = ImprovedCNN(num_classes=len(dataset.classes)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Addestramento del modello
        train_model(model, train_loader, val_loader, num_epochs, file)

        # Salvataggio del modello
        torch.save(model.state_dict(), f"{data_dir}_gasf_classifier_batch_{batch_size}.pth")
        print(f"Modello salvato come '{data_dir}_gasf_classifier_batch_{batch_size}.pth'.")
        
        # Testing
        predictions, precision, class_total, class_correct = test_on_images_folder_with_subfolders_and_precision(model, test_folder, transform, dataset.classes, file)
python3 NetDiffus_Old/scripts/image_train.py --data_dir /home/franc_ubuntu/Università/Progetto/materiale/dataset/Mirage-AppxActPadding.parquet --image_size 10 --num_channels 128 --num_res_blocks 3 --diffusion_steps 100 --noise_schedule cosine --learn_sigma True --class_cond True --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 16