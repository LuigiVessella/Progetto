import numpy as np
import os
from pyts.image import GramianAngularField
import shutil
import matplotlib.pyplot as plt
from skimage.transform import resize

base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "../../128/iterate/df/synth_models/NPZ_Files/samples_1x10x10x3.npz")
output_dir = os.path.join(base_dir, "../../128/iterate/df/synth_models/GASF_SYNT")


# Crea la directory di output se non esiste
os.makedirs(output_dir, exist_ok=True)

# Carica il file npz
npz_data = np.load(input_file_path)
array = npz_data['arr_0']  # Cambia il nome della chiave se necessario

# Inizializza la trasformazione GASF
gasf = GramianAngularField(image_size=10)  # Dimensione dell'immagine

# Loop per ciascun esempio
for idx, sample in enumerate(array):
    # Applica GASF alla sequenza, considerando ogni immagine come un'unica sequenza di valori
    pl_normalized = sample.flatten()  # Flatten della matrice 10x10 (se necessario)
    gasf_image = gasf.fit_transform([pl_normalized])[0]  # Trasformazione GASF

    # Correzione gamma (opzionale, come nel caso delle immagini originali)
    gasf_image = gasf_image * 0.5 + 0.5
    gamma = 0.25
    gasf_image = np.power(gasf_image, gamma)

    # Ridimensiona l'immagine a 10x10
    gasf_image_resized = resize(gasf_image, (10, 10), anti_aliasing=True)

    # Salva come immagine
    img_path = os.path.join(output_dir, f"image_{idx}.png")
    plt.imsave(img_path, gasf_image_resized, cmap='viridis')

print(f"Completato! {len(array)} immagini GASF salvate nella directory {output_dir}.")
