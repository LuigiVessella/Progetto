import pandas as pd
import numpy as np
import os
import imageio  # Importare imageio per il salvataggio delle immagini

from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import imageio
import sys

# Funzione per calcolare il minimo e massimo globali su PL e aggiunta di DIR 
def find_global_min_max_with_dir(df, pl_column, dir_column):
    all_pl_adjusted = []
    for _, row in df.iterrows():
        pl = row[pl_column]
        dir = row[dir_column]
        
        # Gestire il caso in cui dir è un array o una lista
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

# Funzione per normalizzare una serie usando min e max globali tra 0 e 1
def normalize_with_global(series, global_min, global_max):
    print(f"Serie originale da normalizzare:{series}")
    print(f"globalmin:{global_min}")
    print(f"globalmax:{global_max}")

    x = (series - global_min) / (global_max - global_min)
    return x

# Funzione per denormalizzare una serie
def denormalize_with_global(series, global_min, global_max):
    return series * (global_max - global_min) + global_min

# Funzione per calcolare la GASF 
def create_gasf(series):
    n = len(series)
    gasf_matrix = np.zeros((n, n))  # Matrice GASF inizializzata a zero
    for i in range(n):
        for j in range(n):
            # Sommare i due valori e calcolare il coseno dell'angolo
            cos_2phi = np.cos(np.arccos(series[i]) + np.arccos(series[j]))  
            gasf_matrix[i, j] = cos_2phi
    return gasf_matrix

# Funzione per deconvertire la diagonale della matrice GASF
def inverse_gasf_conversion(diagonal):
    # Nuova formula per de-GASFizzare (radice quadrata dell'elemento)
    return np.sqrt((diagonal + 1) / 2)

# Funzione per svuotare la directory di destinazione
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_directory(file_path)
            os.rmdir(file_path)


# Path relativo per il file di input e la directory di output
base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "Mirage-CrunchRollOnly.parquet")

# Directory per salvare le immagini GASF, usando un percorso relativo
output_file_path = os.path.join(base_dir, "GASF_Original_CrunchRoll")

df = pd.read_parquet(input_file_path)

global_min, global_max = find_global_min_max_with_dir(df, pl_column="PL", dir_column="DIR")

# Creare una cartella per salvare le immagini GASF
os.makedirs(output_file_path, exist_ok=True)

# Svuotare la directory prima di salvare nuove immagini
clear_directory(output_file_path)


# Elaborare tutte le righe del dataset
for idx, row in df.iterrows():
    pl = np.array(row["PL"])
    normalized_series = normalize_with_global(pl, global_min, global_max)
    gasf = create_gasf(normalized_series)

    # Normalizzare la matrice GASF per essere salvata come immagine in scala di grigi (1 canale)
    gasf_scaled = ((gasf - gasf.min()) / (gasf.max() - gasf.min()) * 255).astype(np.uint8)

    # Creare una cartella per l'etichetta (LABEL)
    label = row["LABEL"]
    label_folder = os.path.join(output_file_path, str(label))
    os.makedirs(label_folder, exist_ok=True)

    # Salvare l'immagine come immagine in scala di grigi (1 canale)
    output_png_path = os.path.join(label_folder, f"GASF_row_{idx}.png")
    imageio.imwrite(output_png_path, gasf_scaled)  # Salvataggio come immagine in scala di grigi
    print(f"Immagine GASF salvata per la riga {idx} in '{output_png_path}'.")
