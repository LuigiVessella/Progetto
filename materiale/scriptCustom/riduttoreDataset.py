#riduce il dataset in una sottoporzione
import os
import pandas as pd

# Percorso del file .parquet
#input_file_path = 'progetto/materiale/Mirage-AppxAct.parquet'  # Modifica con il percorso del tuo file

# Ottieni la directory corrente dello script
base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "../Mirage-AppxAct.parquet")
output_file_path = os.path.join(base_dir, "../Mirage-AppxActRidotto.parquet")
data = pd.read_parquet(input_file_path)

half_data = data.iloc[:len(data)//100]  

# Salva il nuovo dataset dimezzato in un file .parquet
half_data.to_parquet(output_file_path)

print(f"Dataset ridotto salvato in: {output_file_path}")
