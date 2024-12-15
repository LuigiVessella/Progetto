import numpy as np
import imageio
import os

# Funzione per deconvertire la diagonale della matrice GASF
def inverse_gasf_conversion(diagonal):
    return np.sqrt((diagonal + 1) / 2)

# Funzione per denormalizzare una serie
def denormalize_with_global(series, global_min, global_max):
    return series * (global_max - global_min) + global_min

# Funzione per caricare l'immagine GASF e recuperare la serie temporale
def recover_series_from_gasf_image(image_path, global_min, global_max):
    # Carica l'immagine GASF in scala di grigi
    grayscale_image = imageio.imread(image_path)
    
    # Verifica se l'immagine è in formato RGB (3 canali) e prendi il canale grigio
    if grayscale_image.ndim == 3:
        grayscale_image = grayscale_image[:, :, 0]  # Prendi solo il primo canale (assumendo che sia in scala di grigi)
    
    # Normalizza l'immagine tra -1 e 1 (ipotizzando che i valori dell'immagine siano tra 0 e 255)
    gasf_matrix = grayscale_image / 255.0 * 2 - 1
    
    # Recupera la diagonale della matrice GASF
    gasf_diagonal = np.diagonal(gasf_matrix)

    # De-GASFizza per ottenere i valori normalizzati dalla diagonale
    recovered_normalized_values = inverse_gasf_conversion(gasf_diagonal)

    # Denormalizza per ottenere i valori originali
    recovered_original_values = denormalize_with_global(recovered_normalized_values, global_min, global_max)
    
    return recovered_original_values

# Percorso dell'immagine GASF salvata come PNG
image_path = "/Users/luigivessella/Desktop/Università/data analitycs/Progetto/materiale/datiOriginali_GASF/Crunchyroll/GASF_row_0.png"  # Sostituisci con il percorso dell'immagine GASF

# Minimo e massimo globali (sostituisci con i valori reali)
global_min = -1448  # Sostituisci con il valore reale
global_max = 1448   # Sostituisci con il valore reale

# Recupera la serie temporale dalla GASF
recovered_series = recover_series_from_gasf_image(image_path, global_min, global_max)

# Stampa la serie recuperata
print("Serie temporale recuperata:", recovered_series)
