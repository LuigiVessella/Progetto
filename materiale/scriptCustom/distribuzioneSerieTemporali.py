import os
import pandas as pd
import matplotlib.pyplot as plt

# Percorsi dei file parquet
base_dir = os.path.dirname(__file__)
file1_path = os.path.join(base_dir,"../dataset/Mirage-AppxActPadding.parquet") #path del dataset originale
file2_path = os.path.join(base_dir,"../dataset/diagonaliSinteticheConvertite/diagonali_sintetiche.parquet") #path del dataset sintetico

# Percorso per la cartella Grafici
base_output_dir = os.path.join(os.path.dirname(file1_path), "GraficiComparativiOriginale_Sintetico")
os.makedirs(base_output_dir, exist_ok=True)

# Carica i file parquet in DataFrame
file1_df = pd.read_parquet(file1_path)
file2_df = pd.read_parquet(file2_path)

# Funzione per creare un grafico per il conteggio dei dati
def plot_histogram(data, label, title, output_path):
    plt.figure(figsize=(12, 6))
    # Creazione dell'istogramma con il conteggio delle occorrenze
    plt.hist(data, bins=30, alpha=0.7, color='blue', label=label)
    
    plt.title(title, fontsize=14)
    plt.xlabel("PL", fontsize=12)
    plt.ylabel("Conteggio", fontsize=12)
    plt.legend(title="Tipo di Dati", labels=[label])

    # Salva il grafico
    plt.savefig(output_path)
    plt.close()

# Creare grafici separati per ciascuna classe (sia per i dati originali che sintetici)
unique_labels = set(file1_df["LABEL"]).union(set(file2_df["LABEL"]))

for label in unique_labels:
    # Estrazione dei dati per la classe
    file1_label_data = file1_df[file1_df["LABEL"] == label]["PL"].explode().tolist()
    file2_label_data = file2_df[file2_df["LABEL"] == label]["PL"].explode().tolist()

    if file1_label_data and file2_label_data:
        # Percorso per la sottocartella della classe
        label_output_dir = os.path.join(base_output_dir, str(label))
        os.makedirs(label_output_dir, exist_ok=True)

        # Grafico per il dataset originale (File1)
        file1_output_path = os.path.join(label_output_dir, f"Confronto_Originale_{label}.png")
        plot_histogram(
            data=file1_label_data,
            label="Dati Originali",
            title=f"Distribuzione PL per la classe '{label}' - Dati Originali",
            output_path=file1_output_path
        )

        # Grafico per il dataset sintetico (File2)
        file2_output_path = os.path.join(label_output_dir, f"Confronto_Sintetico_{label}.png")
        plot_histogram(
            data=file2_label_data,
            label="Dati Sintetici",
            title=f"Distribuzione PL per la classe '{label}' - Dati Sintetici",
            output_path=file2_output_path
        )

# Grafico comparativo per l'intero dataset (totale, senza distinzione per classe)
file1_all_data = file1_df["PL"].explode().tolist()
file2_all_data = file2_df["PL"].explode().tolist()

# Percorso per la cartella globale
global_output_dir = os.path.join(base_output_dir, "Globale")
os.makedirs(global_output_dir, exist_ok=True)

# Grafico per il dataset originale (File1)
file1_global_output_path = os.path.join(global_output_dir, "Confronto_Originale_Globale.png")
plot_histogram(
    data=file1_all_data,
    label="Dati Originali",
    title="Distribuzione PL - Dati Originali (Globale)",
    output_path=file1_global_output_path
)

# Grafico per il dataset sintetico (File2)
file2_global_output_path = os.path.join(global_output_dir, "Confronto_Sintetico_Globale.png")
plot_histogram(
    data=file2_all_data,
    label="Dati Sintetici",
    title="Distribuzione PL - Dati Sintetici (Globale)",
    output_path=file2_global_output_path
)
