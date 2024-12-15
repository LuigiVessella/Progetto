#script creato per verificare se il salvataggio delle GASF del dataset originale venivano mantenute nel npz e quindi riconvertite correttamente in serie temporale
import pandas as pd
import numpy as np
import os

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
input_file_path = "/home/franc_ubuntu/Università/Progetto/materiale/dataset/Mirage-AppxActPadding.parquet"
output_file_path = "/home/franc_ubuntu/Università/Progetto/materiale/dataset/datasetOriginaleNPZ"

df = pd.read_parquet(input_file_path)

# Calcolo del global_min e global_max per l'intera colonna PL
all_pl_values = [value for pl_list in df["PL"] for value in pl_list]
global_min, global_max = min(all_pl_values), max(all_pl_values)

# Creare una cartella per salvare le immagini GASF
os.makedirs(output_file_path, exist_ok=True)

# Dizionario per raccogliere le immagini per ciascuna classe
class_images = {}

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

    # Recupera l'etichetta (LABEL)
    label = row["LABEL"]

    # Crea un array per la classe, se non esiste già
    if label not in class_images:
        class_images[label] = []

    # Aggiungi l'immagine alla lista corrispondente alla classe
    gasf_resized_with_channel = gasf_resized.reshape((10, 10, 1))  # Forma (10, 10, 1)
    class_images[label].append(gasf_resized_with_channel)

# Ora salviamo i file .npz per ogni classe
for label, images in class_images.items():
    # Converti la lista di immagini in un array numpy
    images_array = np.array(images)

    # Salva l'array in un file .npz
    npz_filename = os.path.join(output_file_path, f"{label}.npz")
    np.savez(npz_filename, images=images_array)

    print(f"File {npz_filename} salvato con {len(images)} immagini.")

print("Conversione completata per tutte le classi.")
