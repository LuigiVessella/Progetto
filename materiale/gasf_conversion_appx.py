import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from skimage.transform import resize

base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "Mirage-AppxActRidotto.parquet")

# Directory per salvare le immagini GASF, usando un percorso relativo
output_file_path = os.path.join(base_dir, "datiOriginali_GASF/")


if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# Leggi il file Parquet
df = pd.read_parquet(input_file_path)

# Verifica che la colonna 'PL' esista
if 'PL' not in df.columns:
    raise ValueError(f"Errore: la colonna 'PL' non è presente nel file {input_file_path}.")

# Processa ogni riga nella colonna PL
k = 1  # Contatore per le immagini generate
for idx, pl_values in df['PL'].items():
    # Verifica che ogni riga contenga 10 elementi
    if len(pl_values) != 10:
        print(f"Errore: la riga {idx} non contiene 10 elementi nella colonna PL. Ignorata.")
        continue

    # Converti i dati in un array NumPy
    points = np.array([pl_values])

    # Calcola il Gramian Angular Field
    gasf = GramianAngularField(sample_range=(0, 1), method='summation')
    X_gasf = gasf.fit_transform(points)

    # Correzione gamma
    gasf_img = X_gasf[0] * 0.5 + 0.5
    gamma = 0.25
    gasf_img = np.power(gasf_img, gamma)

    # Ridimensiona l'immagine a 128x128
    gasf_img_resized = resize(gasf_img, (128, 128), anti_aliasing=True)

    # Salva come PNG
    image_name = f"Mirage_PL_{k}.png"
    plt.imsave(os.path.join(output_file_path, image_name), gasf_img_resized, cmap='viridis')

    k += 1

print(f"Completato! {k - 1} immagini GASF salvate nella directory {output_file_path}.")
