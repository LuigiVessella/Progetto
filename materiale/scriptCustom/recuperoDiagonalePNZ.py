import pandas as pd
import numpy as np
import os
import imageio  # Importare imageio per il salvataggio delle immagini
import matplotlib.pyplot as plt  # Importare matplotlib per applicare colormap



input_file_path = "/home/franc_ubuntu/Università/Progetto/materiale/dataset/Mirage-AppxActPadding.parquet"
df = pd.read_parquet(input_file_path)

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

global_min, global_max = find_global_min_max_with_dir(df, pl_column="PL", dir_column="DIR")

def deGASF(diagonale):
    return np.sqrt((diagonale + 1)/2)

def deNorm(diagonaleDeGASF):
    return diagonaleDeGASF * (global_max - global_min) + global_min

npz_path = "/home/franc_ubuntu/Università/Progetto/materiale/dataset/datasetConNPZ/Crunchyroll/Crunchyroll_0.npz"
gasf_from_npz = np.load(npz_path)['gasf']
npz_diag = np.diagonal(gasf_from_npz)
print(f"Diagonale della GASF {npz_diag}")
diagonaleDeGASF = deGASF(npz_diag)
print(f"Diagonale de GASF {diagonaleDeGASF}")
diagonaleDeNorm = deNorm(diagonaleDeGASF)
print(f"Diagonale principale della GASF dal file NPZ: {diagonaleDeNorm.astype(int)}")

