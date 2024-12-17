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
input_dir = os.path.join(base_dir, "../../NetDiffus/128/iterate/df/synth_models/immaginiSinteticheNPZ")
output_parquet = os.path.join(base_dir, "../dataset/1")
os.makedirs(output_parquet, exist_ok=True)
output_parquet_path = os.path.join(output_parquet, "1_test_original.parquet")

# Carica il dataset per calcolare global_min e global_max
input_file_path = "/Users/luigivessella/Desktop/Università/data analitycs/Progetto/materiale/dataset/Mirage-AppxActPadding.parquet"
input_file_path = os.path.join(base_dir, "../dataset/Mirage-AppxActPadding.parquet")
df_global = pd.read_parquet(input_file_path)

all_pl_values = []
for _, row in df_global.iterrows():
    pl = np.array(row["PL"])
    dir = np.array(row["DIR"])

    # Aggiusta il segno di PL in base a DIR
    pl_adjusted = np.array([p if d != 0 else -p for p, d in zip(pl, dir)])
    all_pl_values.extend(pl_adjusted)

global_min, global_max = min(all_pl_values), max(all_pl_values)
# Calcolo global_min e global_max
print(global_min, global_max)

# Inizializzazione DataFrame vuoto
result_df = pd.DataFrame(columns=["PL", "DIR", "LABEL"])

# Sostituisci con il nome del file NPZ che vuoi testare
file_name = ""  # Modifica qui con il nome del file che vuoi testare


# Carica il file NPZ
data = np.load("/Users/luigivessella/Desktop/Università/data analitycs/Progetto/materiale/dataset/datasetTestInNPZ/Crunchyroll/Crunchyroll_1.npz")

# Carica la matrice gasf dalla chiave "gasf"
gasf_matrix = data["gasf"]

# Verifica se la matrice è 2D
if gasf_matrix.ndim == 2:
    # Estrai la diagonale principale
    diagonale = np.diagonal(gasf_matrix)
else:
    print(f"Errore: la matrice non è 2D. Forma matrice: {gasf_matrix.shape}")
    diagonale = []

# Se la diagonale è stata estratta, continua il processo
if diagonale.size > 0:
    # Applica deGASF e deNorm
    diagonale_de_gasf = deGASF(diagonale)
    diagonale_de_norm = deNorm(diagonale_de_gasf, global_min, global_max)

    # Arrotonda i valori di PL (senza decimali)
   

    
    dir_values = [1 if value >= 0 else 0 for value in diagonale_de_norm]
    
    diagonale_de_norm = [abs(round(value)) for value in diagonale_de_norm]


    # Aggiungi al DataFrame (solo una riga)
    result_df = pd.concat(
        [
            result_df,
            pd.DataFrame(
                {
                    "PL": [diagonale_de_norm],
                    "DIR": [dir_values],
                    "LABEL": [file_name.replace(".npz", "")]
                }
            )
        ],
        ignore_index=True
    )

    # Salva il DataFrame in formato Parquet
    result_df.to_parquet(output_parquet_path, index=False)

    print(f"File Parquet salvato in: {output_parquet_path}")
else:
    print(f"Impossibile estrarre la diagonale dalla matrice. Forma matrice: {gasf_matrix.shape}")
