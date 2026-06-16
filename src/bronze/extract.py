import kagglehub
import pandas as pd
from pathlib import Path
import json

# download kaggle goemotion dataset 
path = kagglehub.dataset_download("debarshichanda/goemotions")

# paths of the data needed from GoEmotions 

dir_path = Path(path)

data_path = Path("data")
full_dataset_path = dir_path / data_path / "full_dataset"
csv1_path = full_dataset_path / "goemotions_1.csv" 
csv2_path = full_dataset_path / "goemotions_2.csv"
csv3_path = full_dataset_path /  "goemotions_3.csv"
sentiment_mapping_path = dir_path / data_path / "sentiment_mapping.json"

# loading json
with open(sentiment_mapping_path) as f:
    sentiment_mapping = json.load(f)

# transforming csv into dfs
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)
df3 = pd.read_csv(csv3_path)

df_raw = pd.concat([df1, df2, df3], ignore_index=True)

# rows, columns = df_raw.shape
# print(f"Rows: {rows}, Columns: {columns}")
# print(df_raw.columns.values.tolist())

