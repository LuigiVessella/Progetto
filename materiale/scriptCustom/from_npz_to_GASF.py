import numpy as np
import os
import imageio

# Funzione per convertire un array RGB in scala di grigi
def convert_to_grayscale(rgb_array):
    # Assumendo che l'array sia già 10x10x3 (o dimensione compatibile)
    return np.mean(rgb_array, axis=-1).astype(np.uint8)

# Funzione per processare un file .npz
def process_npz_to_grayscale_images(npz_path, output_folder):
    try:
        data = np.load(npz_path)

        # Verifica delle chiavi necessarie
        if 'arr_0' not in data or 'arr_1' not in data:
            raise KeyError("File .npz non contiene le chiavi 'arr_0' e 'arr_1'.")

        images = data['arr_0']
        labels = data['arr_1']

        # Verifica della coerenza tra immagini ed etichette
        if len(images) != len(labels):
            raise ValueError("Numero di immagini ed etichette non corrisponde.")

        for idx, (image, label) in enumerate(zip(images, labels)):
            # Verifica che l'immagine sia 10x10x3 (o 10x10)
            if image.shape[:2] != (10, 10):
                raise ValueError(f"L'immagine all'indice {idx} non ha dimensioni 10x10.")

            # Convertire in scala di grigi
            grayscale_image = convert_to_grayscale(image)

            # Creare una cartella per ogni etichetta
            label_folder = os.path.join(output_folder, str(label))
            os.makedirs(label_folder, exist_ok=True)

            # Salvare l'immagine
            output_path = os.path.join(label_folder, f"image_{idx}.png")
            imageio.imwrite(output_path, grayscale_image)
            print(f"Immagine salvata: {output_path}")

    except Exception as e:
        print(f"Errore durante il processamento di {npz_path}: {e}")

# Directory dei file .npz
base_dir = os.path.dirname(__file__)
input_dir = os.path.join(base_dir, "../../128/iterate/df/synth_models/GASF_SYNTH")
output_dir = os.path.join(base_dir, "../../128/iterate/df/synth_models/grayscale_images")

# Creare la directory di output
os.makedirs(output_dir, exist_ok=True)

# Processare i file nella directory di input
for filename in os.listdir(input_dir):
    if filename.endswith(".npz"):
        npz_path = os.path.join(input_dir, filename)
        process_npz_to_grayscale_images(npz_path, output_dir)
