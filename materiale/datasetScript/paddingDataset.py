#lo script va fatto partire dalla cartella che contiene modificaDataset
#vanno installati pandas,  pyarrow e fastparquet
#aggiunge gli 0 nei campi che non contengono 10 valori
import os
import pandas as pd;



base_dir = os.path.dirname(__file__)
input_file_path = os.path.join(base_dir, "../Mirage-AppxAct.parquet")
output_file_path = os.path.join(base_dir, "../Mirage-AppxActPadding.parquet")


df = pd.read_parquet(input_file_path)

def insertDummies(column):
    column = column.tolist()
    print("Modifico la colonna con {len(column)} elementi: {column}")
    while len(column) < 10:
        column.append(0)
    return column

for index, row in df.iterrows(): 
    for col in row.index:
        if col != "LABEL":
            if len(row[col]) < 10:
                df.at[index, col] = insertDummies(row[col])



df.to_parquet(output_file_path)

