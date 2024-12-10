import numpy as np
import imageio
from skimage.transform import resize

# Funzione per la riconversione della matrice GASF alla serie temporale
def restore_series_from_gasf(image_path, num_packets, min_global, max_global):
    # Leggi l'immagine GASF
    gasf_img = imageio.imread(image_path)

    # Assicurati che l'immagine sia 2D e ridimensiona a 10x10 se necessario
    gasf_img_resized = resize(gasf_img, (num_packets, num_packets), anti_aliasing=True)

    # Estrai gli elementi diagonali della matrice (che rappresentano la serie temporale)
    diagonal_elements = np.diag(gasf_img_resized)

    # Applica la formula inversa per recuperare i valori originali
    restored_values = [np.sqrt((val + 1) / 2) for val in diagonal_elements]

    # Denormalizza (se necessario)
    restored_values = [(val * (max_global - min_global)) + min_global for val in restored_values]

    return restored_values

# Esempio di utilizzo:
gasf_image_path = '/Users/luigivessella/Desktop/Università/data analitycs/Progetto/materiale/datiOriginali_GASF/Crunchyroll/Mirage_PL_1.png'  # Percorso dell'immagine GASF
num_packets = 10  # Numero di pacchetti (dimensione della matrice)
min_global = -10  # Minimo globale che hai calcolato durante la normalizzazione
max_global = 10   # Massimo globale che hai calcolato durante la normalizzazione

restored_series = restore_series_from_gasf(gasf_image_path, num_packets, min_global, max_global)

print("Serie temporale riconvertita:", restored_series)
