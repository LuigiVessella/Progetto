import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread

# Funzione per calcolare il minimo e massimo globali su PL e aggiunta di DIR 
def find_global_min_max_with_dir(df, pl_column, dir_column):
    all_pl_adjusted = []
    for _, row in df.iterrows():
        pl = row[pl_column]
        dir = row[dir_column]
        if dir==0:
            pl_inverted = []
            for p in pl:
                pl_inverted.append(-p)
            pl = pl_inverted
        all_pl_adjusted.extend(pl)
    global_min = min(all_pl_adjusted)
    global_max = max(all_pl_adjusted)
    return global_min, global_max

# Funzione per normalizzare una serie usando min e max globali tra 0 e 1
def normalize_with_global(series, global_min, global_max):
    return (series - global_min) / (global_max - global_min)

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

# Path del file parquet 
file_path = os.path.join("materiale", "trainSet.parquet")

df = pd.read_parquet(file_path)

global_min, global_max = find_global_min_max_with_dir(df, pl_column="PL", dir_column="DIR")

# Creare una cartella per salvare le immagini GASF
if(file_path.__contains__('trainSet')):
    output_folder = os.path.join("materiale","immaginiGASFTrain")

else:
    output_folder = os.path.join("materiale","immaginiGASFTest")

os.makedirs(output_folder, exist_ok=True)

# Iterare su tutte le righe del dataset
for idx, row in df.iterrows():
    pl = np.array(row["PL"])
    normalized_series = normalize_with_global(pl, global_min, global_max)
    gasf = create_gasf(normalized_series)

    label = row["LABEL"]
    label_folder = os.path.join(output_folder, str(label))
    os.makedirs(label_folder, exist_ok=True)

    output_path = os.path.join(label_folder, f"GASF_row_{idx}.png")
    plt.imsave(output_path, gasf, cmap='viridis')

print(f"Immagini GASF create e salve in cartella '{output_folder}'.")



# export RDMAV_FORK_SAFE=1
#python3 scripts/image_train.py --data_dir <Cartella immagini GASF originali> --image_size 10 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule cosine --learn_sigma True --class_cond True --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 16
#python3 scripts/image_sample.py --model_path <cartella con modello creato> --image_size 10 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule cosine --learn_sigma True --class_cond True --rescale_learned_sigmas False --rescale_timesteps False