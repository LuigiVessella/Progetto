
'''
example of use:
python materiale/simple_classifier_2.py \
    --train_path materiale/dataset/trainSet.parquet \
    --test_path materiale/dataset/testSet.parquet \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --batch_size 4


'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

# Parsing degli argomenti dalla riga di comando
def parse_args():
    parser = ArgumentParser(description="Train and test an RNN model with Torchmetrics")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument('--train_path', type=str, required=True, help="Path to the training Parquet dataset")
    parser.add_argument('--test_path', type=str, required=True, help="Path to the testing Parquet dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and validation")
    return parser.parse_args()

# Dataset personalizzato
class ParquetDataset(Dataset):
    def __init__(self, data, label_to_idx):
        self.data = data
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pl_series = torch.tensor(row['PL'], dtype=torch.float32)
        label = torch.tensor(self.label_to_idx[row['LABEL']], dtype=torch.long)
        return pl_series, label

# Modello RNN
class RNNModel(nn.Module):
    def __init__(self, input_length, num_classes, hidden_size=128, num_layers=2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_length, 1)
        x = x.unsqueeze(-1)  # Aggiunge dimensione per l'input_size dell'LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Prendiamo l'output dell'ultimo time step
        out = self.fc(out)
        return out

# Funzione di training
def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion, device):
    train_accuracy = MulticlassAccuracy(num_classes=model.fc.out_features, average='macro').to(device)
    val_accuracy = MulticlassAccuracy(num_classes=model.fc.out_features, average='macro').to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        train_acc = 0

        for pl_series, labels in train_loader:
            pl_series, labels = pl_series.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pl_series)
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
            for pl_series, labels in val_loader:
                pl_series, labels = pl_series.to(device), labels.to(device)
                outputs = model(pl_series)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += val_accuracy(outputs, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss={epoch_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# Funzione di testing
def test_model(model, test_loader, num_classes, device):
    precision = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    recall = MulticlassRecall(num_classes=num_classes, average=None).to(device)
    f1_score = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    model.eval()
    with torch.no_grad():
        all_precisions = torch.zeros(num_classes).to(device)
        all_recalls = torch.zeros(num_classes).to(device)
        all_f1s = torch.zeros(num_classes).to(device)

        for pl_series, labels in test_loader:
            pl_series, labels = pl_series.to(device), labels.to(device)
            outputs = model(pl_series)
            preds = torch.argmax(outputs, dim=1)

            all_precisions += precision(preds, labels)
            all_recalls += recall(preds, labels)
            all_f1s += f1_score(preds, labels)

        all_precisions /= len(test_loader)
        all_recalls /= len(test_loader)
        all_f1s /= len(test_loader)

        print("Class-wise Metrics:")
        for i in range(num_classes):
            print(f"Class {i}: Precision={all_precisions[i] * 100:.2f}%, "
                  f"Recall={all_recalls[i] * 100:.2f}%, "
                  f"F1-Score={all_f1s[i] * 100:.2f}%")

# Main
if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caricamento del dataset
    train_df = pd.read_parquet(args.train_path)
    test_df = pd.read_parquet(args.test_path)
    label_to_idx = {label: idx for idx, label in enumerate(train_df['LABEL'].unique())}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Suddivisione training/validation
    train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['LABEL'], random_state=42)

    train_dataset = ParquetDataset(train_data, label_to_idx)
    val_dataset = ParquetDataset(val_data, label_to_idx)
    test_dataset = ParquetDataset(test_df, label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Modello
    input_length = len(train_df.iloc[0]['PL'])
    num_classes = len(label_to_idx)
    model = RNNModel(input_length=input_length, num_classes=num_classes).to(device)

    # Ottimizzatore e funzione di perdita
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training
    train_model(model, train_loader, val_loader, args.num_epochs, optimizer, criterion, device)

    # Testing
    test_model(model, test_loader, num_classes, device)
