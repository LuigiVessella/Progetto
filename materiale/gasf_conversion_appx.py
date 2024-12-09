import shutil
import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from skimage.transform import resize

# Path relativo per il file di input e la directory di output
base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "Mirage-AppxActRidotto.parquet")

# Directory per salvare le immagini GASF, usando un percorso relativo
output_file_path = os.path.join(base_dir, "datiOriginali_GASF")
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)


# Elimina i file esistenti nella directory di output
if os.path.exists(output_file_path):
    shutil.rmtree(output_file_path)  # Rimuove la directory e il suo contenuto
os.makedirs(output_file_path)  # Ristabilisce la directory vuota

# Caricamento del dataset
df = pd.read_parquet(input_file_path)

# Verifica che la colonna 'PL' esista
if 'PL' not in df.columns:
    raise ValueError(f"Errore: la colonna 'PL' non è presente nel file {input_file_path}.")

# Calcolo del minimo e massimo globali per la normalizzazione
all_pl_adjusted = []
for _, row in df.iterrows():
    pl = row['PL']  # Lista delle dimensioni dei pacchetti
    # Creazione dei valori positivi/negativi per PL
    pl_adjusted = [p if d == 1 else -p for p, d in zip(pl, row['DIR'])]
    all_pl_adjusted.extend(pl_adjusted)

# Calcolare i minimi e massimi globali
min_global = min(all_pl_adjusted)
max_global = max(all_pl_adjusted)
print(f"Global Min: {min_global}, Global Max: {max_global}")

# Parametri
num_packets = 10  # Numero massimo di pacchetti per campione

# Processa ogni riga nella colonna PL
k = 1  # Contatore per le immagini generate
for idx, row in df.iterrows():
    pl_values = row['PL']
    label = row['LABEL']  # Nome della classe del campione
    dir_ = row['DIR']  # Lista delle direzioni dei pacchetti (1 o 0)

    # Creazione dei valori positivi/negativi per PL
    pl_adjusted = [p if d == 1 else -p for p, d in zip(pl_values, dir_)]
    
    # Riempimento o troncamento a 10 valori
    pl_adjusted = (pl_adjusted + [np.int64(0)] * num_packets)[:num_packets]

    # Normalizzazione Min-Max globale tra -1 e 1
    if max_global - min_global != 0:  # Evita divisioni per zero
        pl_normalized = [(p - min_global) / (max_global - min_global) for p in pl_adjusted]
    else:
        pl_normalized = [0] * num_packets

    # Calcola il Gramian Angular Field (GASF)
    gasf = GramianAngularField(sample_range=None, method='summation')
    X_gasf = gasf.transform([pl_normalized])

    # Correzione gamma (opzionale)
    gasf_img = X_gasf[0] * 0.5 + 0.5
    gamma = 0.25
    gasf_img = np.power(gasf_img, gamma)

    # Ridimensiona l'immagine a 128x128, ora 10x10
    gasf_img_resized = resize(gasf_img, (10, 10), anti_aliasing=True)

    # Crea la directory per l'etichetta se non esiste
    label_dir = os.path.join(output_file_path, str(label))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Salva l'immagine come PNG
    image_name = f"Mirage_PL_{k}.png"
    image_path = os.path.join(label_dir, image_name)
    plt.imsave(image_path, gasf_img_resized, cmap='viridis')

    k += 1

print(f"Completato! {k - 1} immagini GASF salvate nella directory {output_file_path}.")
