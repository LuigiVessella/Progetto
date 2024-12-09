import os
import numpy as np
import matplotlib.pyplot as plt


base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "../../128/iterate/df/synth_models/GASF_SYNTH/samples_1x10x10x3.npz")

# Simula il caricamento dell'array da un file .npz
npz_file = np.load(input_file_path)
image_data = npz_file['arr_0']

# Rimuove dimensioni inutili (se necessario)
image_data = np.squeeze(image_data)

# Converte i valori in uint8 (necessari per rappresentare un'immagine)
image_data = image_data.astype(np.uint8)

# Salva l'immagine come file PNG
plt.imsave('output_image.png', image_data)

# Mostra l'immagine (opzionale)
plt.imshow(image_data)
plt.axis('off')
plt.show()
