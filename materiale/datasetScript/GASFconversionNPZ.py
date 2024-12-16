import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Funzioni per normalizzare e creare GASF
def normalize_with_global(series, global_min, global_max):
    return (series - global_min) / (global_max - global_min)

def create_gasf(series):
    n = len(series)
    gasf_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_2phi = np.cos(np.arccos(series[i]) + np.arccos(series[j]))
            gasf_matrix[i, j] = cos_2phi
    return gasf_matrix

# Path e impostazioni
base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir,"../dataset/Mirage-AppxActPadding.parquet")
output_file_path = os.path.join(base_dir,"../dataset/datasetOriginaleInNPZ") #da passare come argomento a image_train.py

df = pd.read_parquet(input_file_path)

# Calcolo del global_min e global_max per l'intera colonna PL
all_pl_values = [value for pl_list in df["PL"] for value in pl_list]
global_min, global_max = min(all_pl_values), max(all_pl_values)

# Creare una cartella per salvare le immagini GASF
os.makedirs(output_file_path, exist_ok=True)

# Elaborare tutte le righe del dataset
for idx, row in df.iterrows():
    # Mantieni la serie PL invariata
    pl = np.array(row["PL"])

    # Normalizza la serie
    normalized_series = normalize_with_global(pl, global_min, global_max)

    # Crea la matrice GASF
    gasf = create_gasf(normalized_series)

    # Ridimensiona la matrice GASF a 10x10
    gasf_resized = np.resize(gasf, (10, 10))

    # Creare una cartella per l'etichetta (LABEL)
    label = row["LABEL"]
    label_folder = os.path.join(output_file_path, str(label))
    os.makedirs(label_folder, exist_ok=True)

    # Percorso per salvare il file NPZ
    output_npz_path = os.path.join(label_folder, f"{label}_{idx}.npz")

    # Salva la matrice GASF in formato NPZ
    np.savez(output_npz_path, gasf=gasf_resized)

    print(f"Immagine GASF salvata per la riga {idx} in '{output_npz_path}'.")
