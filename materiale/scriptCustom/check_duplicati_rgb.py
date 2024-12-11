import os
import numpy as np

# Funzione per verificare se i canali sono duplicati
def are_channels_duplicated(rgb_image):
    if rgb_image.shape[-1] != 3:
        raise ValueError("L'immagine non ha 3 canali.")
    # Controlla se i tre canali sono identici
    return np.all(rgb_image[:, :, 0] == rgb_image[:, :, 1]) and np.all(rgb_image[:, :, 1] == rgb_image[:, :, 2])

base_dir = os.path.dirname(__file__)
input_dir = os.path.join(base_dir, "../../128/iterate/df/synth_models/GASF_SYNTH")
# Carica il file .npz
npz_path = input_dir + "/samples_5x10x10x3.npz"
data = np.load(npz_path)

# Verifica delle immagini
images = data['arr_0']
duplicate_channels = []

for idx, img in enumerate(images):
    if img.shape[-1] == 3:  # Verifica solo per immagini RGB
        duplicated = are_channels_duplicated(img)
        duplicate_channels.append((idx, duplicated))

# Stampa il risultato
for idx, is_duplicated in duplicate_channels:
    print(f"Immagine {idx}: {'Duplicati' if is_duplicated else 'Diversi'}")

print(f"Numero totale di immagini con canali duplicati: {sum(1 for _, d in duplicate_channels if d)}")
