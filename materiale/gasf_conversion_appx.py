import pandas as pd
import numpy as np
import os
import imageio  # Importare imageio per il salvataggio delle immagini

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


# Path relativo per il file di input e la directory di output
base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "Mirage-AppxActRidotto.parquet")

# Directory per salvare le immagini GASF, usando un percorso relativo
output_file_path = os.path.join(base_dir, "datiOriginali_GASF")

# Path del file parquet 
#file_path = "/home/francunix/DataAnalytics/NetDiffus/Mirage-AppxActModifiedRidotto.parquet"

df = pd.read_parquet(input_file_path)

global_min, global_max = find_global_min_max_with_dir(df, pl_column="PL", dir_column="DIR")

# Creare una cartella per salvare le immagini GASF
#output_folder = "/home/francunix/DataAnalytics/NetDiffus/immaginiGrigie"
os.makedirs(output_file_path, exist_ok=True)

# Elaborare solo la prima riga del dataset
row = df.iloc[0]
pl = np.array(row["PL"])
normalized_series = normalize_with_global(pl, global_min, global_max)
print(f"Stampa della serie originale normalizzata: {normalized_series}")
gasf = create_gasf(normalized_series)

# Stampa della diagonale della matrice GASF
gasf_diagonal = np.diagonal(gasf)
np.set_printoptions(suppress=True, precision=6)  # Sopprime la notazione scientifica, precisione a 6 cifre
print(f"Diagonale GASF originale: {gasf_diagonal}")

label = row["LABEL"]
label_folder = os.path.join(output_file_path, str(label))
os.makedirs(label_folder, exist_ok=True)

output_path = os.path.join(label_folder, "GASF_row_0.npz")

# Salvataggio della matrice GASF in un file .npz
np.savez(output_path, gasf=gasf)
print(f"Immagine GASF per la prima riga creata e salvata in '{output_path}'.")

# Rileggere la matrice GASF salvata dal file .npz
data = np.load(output_path)
gasf_reloaded = data['gasf']  # Recuperiamo la matrice GASF

# Deconversione della diagonale
gasf_reloaded_diagonal = np.diagonal(gasf_reloaded)
print(f"Diagonale dell'immagine salvata: {gasf_reloaded_diagonal}")

recovered_normalized_values = inverse_gasf_conversion(gasf_reloaded_diagonal)
print(f"Valori deGASfizzati recuperati dalla diagonale: {recovered_normalized_values}")

# Denormalizzazione
recovered_original_values = denormalize_with_global(recovered_normalized_values, global_min, global_max)
print(f"Valori originali recuperati dalla diagonale: {recovered_original_values}")