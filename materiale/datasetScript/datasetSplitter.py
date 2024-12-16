import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Carica il dataset
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path,'../dataset/Mirage-AppxActPadding.parquet')
print(file_path)
data = pd.read_parquet(file_path)

label_column = 'LABEL'

if label_column not in data.columns:
    raise ValueError(f"Colonna '{label_column}' non trovata nel dataset. Controlla il nome della colonna.")

# Dividi il dataset per classe
train_data = []
test_data = []

for label, group in data.groupby(label_column):
    train, test = train_test_split(group, test_size=0.2, random_state=42)
    train_data.append(train)
    test_data.append(test)

# Combina i dati divisi in DataFrame
train_data = pd.concat(train_data).reset_index(drop=True)
test_data = pd.concat(test_data).reset_index(drop=True)

train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)


# Salva i sotto-dataset
train_output_path = os.path.join(base_path,'../dataset/trainSet.parquet')
test_output_path = os.path.join(base_path,'../dataset/testSet.parquet')

train_data.to_parquet(train_output_path, index=False)
test_data.to_parquet(test_output_path, index=False)

print(f"Dataset di training salvato in: {train_output_path}")
print(f"Dataset di test salvato in: {test_output_path}")
