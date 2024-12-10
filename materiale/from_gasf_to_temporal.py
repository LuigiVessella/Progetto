import numpy as np
import imageio.v3 as iio
from pyts.image import GramianAngularField
import os

# Percorso della directory GASF e del file originale
base_dir = os.path.dirname(__file__)
gasf_dir = os.path.join(base_dir, "datiOriginali_GASF")
output_file_path = os.path.join(base_dir, "Mirage-GeneratedSeries")
print(f"Percorso GASF: {gasf_dir}")
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# Parametri
num_packets = 10  # Numero massimo di pacchetti per campione

# Funzione per rigenerare una serie temporale
def gasf_to_time_series(gasf_image, min_global, max_global):
    """
    Ritorna la serie temporale dalla matrice GASF.
    :param gasf_image: matrice GASF normalizzata (NxN)
    :param min_global: minimo globale utilizzato per la normalizzazione
    :param max_global: massimo globale utilizzato per la normalizzazione
    :return: serie temporale rigenerata
    """
    # Ripristina i valori angolari dal GASF
    gasf_inverse = np.arccos(gasf_image)

    # Ottieni i valori normalizzati dalla diagonale principale
    time_series_normalized = np.cos(np.diag(gasf_inverse))

    # Ripristina i valori originali utilizzando il range globale
    time_series = time_series_normalized * (max_global - min_global) + min_global

    return time_series

# Itera sui file GASF
for label_dir in os.listdir(gasf_dir):
    label_path = os.path.join(gasf_dir, label_dir)
    if not os.path.isdir(label_path):
        print('ciao')
        continue

    for gasf_file in os.listdir(label_path):
        print('for')
        if not gasf_file.endswith(".png"):
            print('if')
            continue

        # Leggi l'immagine GASF
        gasf_path = os.path.join(label_path, gasf_file)
        gasf_image = iio.imread(gasf_path).astype(np.float32)

        # Normalizza i valori GASF tra -1 e 1
        gasf_image = gasf_image / 255 * 2 - 1

        # Ricostruisci la serie temporale
        time_series = gasf_to_time_series(gasf_image, min_global=-1, max_global=1)

        print(time_series)

        # Salva la serie temporale rigenerata
        series_output_path = os.path.join(output_file_path, gasf_file.replace(".png", ".txt"))
        np.savetxt(series_output_path, time_series, fmt="%.6f")

        print(f"Serie temporale salvata: {series_output_path}")

print(f"Completato! Le serie temporali sono state rigenerate nella directory {output_file_path}.")
