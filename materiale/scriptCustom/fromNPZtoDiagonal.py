import os
import numpy as np
import pandas as pd

# Funzioni per deGASF e deNORM
def deGASF(diagonale):
    return np.sqrt((diagonale + 1) / 2)

def deNorm(diagonale_de_gasf, global_min, global_max):
    return diagonale_de_gasf * (global_max - global_min) + global_min

# Percorsi
base_dir = os.path.dirname(__file__)
input_dir = "/Users/luigivessella/Desktop/Università/data analitycs/Progetto/old_material/dataset/datasetOriginaleNPZ"
output_parquet = "/Users/luigivessella/Desktop/Università/data analitycs/Progetto/materiale/dataset/diagonaliOriginaliConvertite"

os.makedirs(output_parquet, exist_ok=True)
output_parquet_path = os.path.join(output_parquet, "diagonali_sintetiche.parquet")

# Calcolo global_min e global_max
input_file_path = "/Users/luigivessella/Desktop/Università/data analitycs/Progetto/materiale/dataset/Mirage-AppxActPadding.parquet"
df_global = pd.read_parquet(input_file_path)

# def find_global_min_max_with_dir(df, pl_column, dir_column):
#     all_pl_adjusted = []
#     for _, row in df.iterrows():
#         pl = row[pl_column]
#         dir = row[dir_column]

#         dir_value = dir[0] if isinstance(dir, (np.ndarray, list)) else dir
#         if dir_value == 0:
#             pl = [-p for p in pl]

#         all_pl_adjusted.extend(pl)

#     global_min = min(all_pl_adjusted)
#     global_max = max(all_pl_adjusted)
#     return global_min, global_max
def find_global_min_max(df, pl_column):
    all_pl_adjusted = []
    for _, row in df.iterrows():
        pl = row[pl_column]
        all_pl_adjusted.extend(pl)

    global_min = min(all_pl_adjusted)
    global_max = max(all_pl_adjusted)
    return global_min, global_max

#global_min, global_max = find_global_min_max_with_dir(df_global, pl_column="PL", dir_column="DIR")
global_min, global_max = find_global_min_max(df_global, pl_column="PL")


# Inizializzazione DataFrame vuoto
result_df = pd.DataFrame(columns=["PL", "DIR", "LABEL"])

# Elaborazione file NPZ
for file_name in os.listdir(input_dir):
    if file_name.endswith(".npz"):
        class_label = file_name.split("_")[0]  # Estrai la classe dal nome del file
        class_label = class_label.replace(".npz", "")  # Rimuovi il suffisso ".npz"
        file_path = os.path.join(input_dir, file_name)

        # Carica il file NPZ
        data = np.load(file_path)["images.npy"]

        # Itera su ogni matrice 10x10
        for idx, matrix in enumerate(data):
            # Estrai la diagonale principale
            diagonale = np.diagonal(matrix[:, :, 0])  # La matrice è 3D con profondità 1

            # Applica deGASF e deNorm
            diagonale_de_gasf = deGASF(diagonale)
            diagonale_de_norm = deNorm(diagonale_de_gasf, global_min, global_max)

            # Arrotonda i valori di PL (senza decimali)
            diagonale_de_norm = [round(value) for value in diagonale_de_norm]

            # Crea DIR basato sui valori di PL
            #dir_values = [1 if value >= 0 else 0 for value in diagonale_de_norm]

            # Rimuovi il segno meno dai valori di PL
            diagonale_de_norm = [abs(value) for value in diagonale_de_norm]

            # Aggiungi al DataFrame
            result_df = pd.concat(
                [
                    result_df,
                    pd.DataFrame(
                        {
                            "PL": [diagonale_de_norm],
                            #"DIR": [dir_values],
                            "LABEL": [class_label]
                        }
                    )
                ],
                ignore_index=True
            )

# Salva il DataFrame in formato Parquet
result_df.to_parquet(output_parquet_path, index=False)

print(f"File Parquet salvato in: {output_parquet_path}")
