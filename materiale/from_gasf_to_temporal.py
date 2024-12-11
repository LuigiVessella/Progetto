import numpy as np
import os
import imageio

# Funzione per deconvertire la diagonale della matrice GASF
def inverse_gasf_conversion(diagonal):
    return np.sqrt((diagonal + 1) / 2)

# Funzione per denormalizzare una serie
def denormalize_with_global(series, global_min, global_max):
    return series * (global_max - global_min) + global_min

# Directory delle immagini sintetiche
base_dir = os.path.dirname(__file__)
input_dir = os.path.join(base_dir, "../128/iterate/df/synth_models/grayscale_images")
output_file = os.path.join(base_dir, "../128/iterate/df/synth_models/recovered/recovered_series.csv")

# Minimo e massimo globali (sostituire con i valori corretti derivati dall'addestramento originale)
global_min = -1448  # Sostituisci con il valore reale
global_max = 1448   # Sostituisci con il valore reale

# Elaborazione delle immagini sintetiche
recovered_data = []

for label_folder in os.listdir(input_dir):
    label_path = os.path.join(input_dir, label_folder)
    if not os.path.isdir(label_path):
        continue

    for filename in os.listdir(label_path):
        if filename.endswith(".png"):
            image_path = os.path.join(label_path, filename)
            
            # Caricare l'immagine in scala di grigi
            grayscale_image = imageio.imread(image_path)
            
            # Normalizzare i valori tra -1 e 1 (ipotesi: valori originali normalizzati)
            gasf_matrix = grayscale_image / 255.0 * 2 - 1
            
            # Recuperare la diagonale della matrice GASF
            gasf_diagonal = np.diagonal(gasf_matrix)

            # De-GASFizzare per ottenere i valori normalizzati
            recovered_normalized_values = inverse_gasf_conversion(gasf_diagonal)

            # Denormalizzare per ottenere i valori originali
            recovered_original_values = denormalize_with_global(recovered_normalized_values, global_min, global_max)

            # Salvare i valori recuperati
            recovered_data.append({
                "label": label_folder,
                "filename": filename,
                "recovered_series": recovered_original_values.tolist()
            })

            print(f"Processata immagine: {filename} | Serie recuperata: {recovered_original_values}")

# Salvare i dati recuperati in un file CSV
import pandas as pd

recovered_df = pd.DataFrame(recovered_data)
recovered_df.to_csv(output_file, index=False)

print(f"Serie temporali recuperate salvate in '{output_file}'")
