import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Funzione per creare istogrammi
def create_histogram(data, label, output_path):
    plt.figure()
    plt.hist(data.ravel(), bins=256, color='blue', alpha=0.7, label=f"{label}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# Funzione principale
def main(original_dir, synthetic_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Creazione delle sottocartelle
    general_plots = os.path.join(output_dir, "GraficiGenerali")
    os.makedirs(general_plots, exist_ok=True)

    # Mappa delle classi: associa nomi originali a numeri sintetici
    class_map = {
        "ClashRoyale": "ClashRoyale", "Crunchyroll": "Crunchyroll", "Discord": "Discord", "JitsiMeet": "JitsiMeet",
        "KakaoTalk": "KakaoTalk", "Line": "Line", "Meets": "Meets", "Omlet": "Omlet",
        "Signal": "Signal", "Slack": "Slack", "Telegram": "Telegram", "Trueconf": "Trueconf",
        "Twitch": "Twitch", "Whatsapp": "Whatsapp"
    }

    # Funzione per caricare immagini da una cartella
    def load_images_from_folder(folder):
        images = []
        for img_path in Path(folder).glob("*.png"):
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            images.append(np.array(img))
        return np.array(images)

    # Caricamento immagini originali e sintetiche
    original_images = {}
    synthetic_images = {}

    for original_label, synthetic_label in class_map.items():
        original_path = Path(original_dir) / original_label
        synthetic_path = Path(synthetic_dir) / synthetic_label

        if not original_path.exists() :
            print(f"Attenzione: la cartella per la classe '{original_label}' non esiste in uno dei dataset originale.")
            continue
        if not synthetic_path.exists():
            print(f"Attenzione: la cartella per la classe '{original_label}' non esiste in uno dei dataset sintetico.")
            continue

        original_images[original_label] = load_images_from_folder(original_path)
        synthetic_images[original_label] = load_images_from_folder(synthetic_path)

        if len(original_images[original_label]) == 0:
            print(f"Attenzione: la cartella originale per la classe '{original_label}' è vuota.")
        if len(synthetic_images[original_label]) == 0:
            print(f"Attenzione: la cartella sintetica per la classe '{synthetic_label}' è vuota.")

    # Creazione istogrammi generali
    try:
        all_original = np.concatenate([img for imgs in original_images.values() for img in imgs], axis=0)
        all_synthetic = np.concatenate([img for imgs in synthetic_images.values() for img in imgs], axis=0)

        create_histogram(all_original, "Original Data", os.path.join(general_plots, "original_data_histogram.png"))
        create_histogram(all_synthetic, "Synthetic Data", os.path.join(general_plots, "synthetic_data_histogram.png"))
    except ValueError:
        print("Errore: impossibile creare istogrammi generali. Verifica che i dati originali e sintetici non siano vuoti.")

    # Creazione grafici per classe
    for original_label, synthetic_label in class_map.items():
        if original_label not in original_images or original_label not in synthetic_images:
            continue

        if len(original_images[original_label]) == 0 or len(synthetic_images[original_label]) == 0:
            continue

        class_dir = os.path.join(output_dir, f"Grafici_{original_label}")
        os.makedirs(class_dir, exist_ok=True)

        create_histogram(original_images[original_label], f"Original {original_label}", os.path.join(class_dir, f"original_{original_label}_histogram.png"))
        create_histogram(synthetic_images[original_label], f"Synthetic {original_label}", os.path.join(class_dir, f"synthetic_{original_label}_histogram.png"))

# Esempio di utilizzo
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    original_dir = os.path.join(base_dir,"../dataset/interoDatasetGASF_RGB")
    synthetic_dir = os.path.join(base_dir,"../dataset/1400ImmaginiSinteticheModello30")
    output_dir = os.path.join(base_dir,"../dataset/graficiModello30k")
    main(original_dir, synthetic_dir, output_dir)
