import numpy as np
import matplotlib.pyplot as plt

# Carica il file .npz
file_path = "/Users/luigivessella/Desktop/Università/data analitycs/Progetto/materiale/dataset/datasetOriginaleInNPZ/Crunchyroll/Crunchyroll_0.npz"  # Sostituisci con il percorso del tuo file
data = np.load(file_path)

# Estrai la matrice 'gasf' dal file
gasf = data['gasf']

# Normalizza i valori per l'intervallo [0, 1] (opzionale, utile per la visualizzazione)
gasf_normalized = (gasf - np.min(gasf)) / (np.max(gasf) - np.min(gasf))

# Visualizza l'immagine
plt.imshow(gasf_normalized, cmap="gray")
plt.colorbar()  # Mostra una barra del colore per riferimento
plt.title("GASF - Immagine in scala di grigi")
plt.axis("off")  # Nasconde gli assi
plt.show()

# Salva l'immagine
output_image_path = "gasf_image.png"
plt.imsave(output_image_path, gasf_normalized, cmap="gray")
print(f"Immagine salvata come {output_image_path}")
