import pandas as pd
import numpy as np
import os
import imageio  # Importare imageio per il salvataggio delle immagini
import matplotlib.pyplot as plt  # Importare matplotlib per applicare colormap

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
    x = (series - global_min) / (global_max - global_min)
    return x

# Funzione per calcolare la GASF 
def create_gasf(series):
    n = len(series)
    gasf_matrix = np.zeros((n, n))  # Matrice GASF inizializzata a zero
    for i in range(n):
        for j in range(n):
            cos_2phi = np.cos(np.arccos(series[i]) + np.arccos(series[j]))  
            gasf_matrix[i, j] = cos_2phi
    return gasf_matrix

# Funzione per deconvertire la diagonale della matrice GASF
def inverse_gasf_conversion(diagonal):
    return np.sqrt((diagonal + 1) / 2)

# Path relativo per il file di input e la directory di output
base_dir = os.path.dirname(__file__)
input_file_path = "/home/franc_ubuntu/Università/Progetto/materiale/dataset/trainSet.parquet"

# Directory per salvare le immagini GASF, usando un percorso relativo
output_file_path = "/home/franc_ubuntu/Università/Progetto/materiale/dataset/trainSetRGB"

df = pd.read_parquet(input_file_path)

global_min, global_max = find_global_min_max_with_dir(df, pl_column="PL", dir_column="DIR")

# Creare una cartella per salvare le immagini GASF
os.makedirs(output_file_path, exist_ok=True)

# Elaborare tutte le righe del dataset
for idx, row in df.iterrows():
    pl = np.array(row["PL"])
    normalized_series = normalize_with_global(pl, global_min, global_max)
    gasf = create_gasf(normalized_series)

    # Ridimensionare la matrice GASF a 10x10
    gasf_resized = np.resize(gasf, (10, 10))

    # Applicare una colormap (ad esempio 'jet') per colorare l'immagine
    gasf_colored = plt.cm.jet((gasf_resized - gasf_resized.min()) / (gasf_resized.max() - gasf_resized.min()))

    # Convertire la matrice colorata in formato RGB (rimuovendo l'alfa)
    gasf_rgb = (gasf_colored[:, :, :3] * 255).astype(np.uint8)

    # Creare una cartella per l'etichetta (LABEL)
    label = row["LABEL"]
    label_folder = os.path.join(output_file_path, str(label))
    os.makedirs(label_folder, exist_ok=True)

    # Salvare l'immagine PNG come RGB
    output_png_path = os.path.join(label_folder, f"{label}_{idx}.png")
    imageio.imwrite(output_png_path, gasf_rgb)

    print(f"Immagine GASF salvata per la riga {idx} in '{output_png_path}'.")

