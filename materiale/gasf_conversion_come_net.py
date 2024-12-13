import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

# Path relativo per il file di input e la directory di output
base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "trainSet.parquet")
output_file_path = os.path.join(base_dir, "trainSet_GASF")

# Carica il file parquet
df = pd.read_parquet(input_file_path)

# Funzione per calcolare il minimo e massimo globali su PL e DIR
def find_global_min_max_with_dir(df, pl_column, dir_column):
    all_pl_adjusted = []
    for _, row in df.iterrows():
        pl = row[pl_column]
        dir = row[dir_column]

        if isinstance(dir, (np.ndarray, list)):
            dir_value = dir[0]
        else:
            dir_value = dir

        if dir_value == 0:
            pl = [-p for p in pl]

        all_pl_adjusted.extend(pl)

    global_min = min(all_pl_adjusted)
    global_max = max(all_pl_adjusted)
    return global_min, global_max

# Normalizzazione della serie
def normalize_with_global(series, global_min, global_max):
    return (series - global_min) / (global_max - global_min)

# Creazione delle immagini GASF
def create_gasf(series):
    gasf = GramianAngularField(method='summation')
    series_reshaped = series.reshape(1, -1)  # PyTS richiede un input 2D
    return gasf.fit_transform(series_reshaped)[0]

# Trova i valori globali minimi e massimi
global_min, global_max = find_global_min_max_with_dir(df, pl_column="PL", dir_column="DIR")

# Creare una cartella per salvare le immagini GASF
os.makedirs(output_file_path, exist_ok=True)

# Elaborare ogni riga del dataset
for idx, row in df.iterrows():
    pl = np.array(row["PL"])
    normalized_series = normalize_with_global(pl, global_min, global_max)
    gasf = create_gasf(normalized_series)

    # Creare una cartella per l'etichetta (LABEL)
    label = row["LABEL"]
    label_folder = os.path.join(output_file_path, str(label))
    os.makedirs(label_folder, exist_ok=True)

    # Percorso di salvataggio
    output_png_path = os.path.join(label_folder, f"GASF_row_{idx}.png")

    # Salvare l'immagine GASF con colormap viridis
    plt.imsave(output_png_path, gasf, cmap='viridis')

    print(f"Immagine GASF salvata per la riga {idx} in '{output_png_path}'.")
