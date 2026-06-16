import pandas as pd

emotioNL_df = pd.read_csv('data/raw/corpus_fulltext_captions.txt', sep="\t")

print(emotioNL_df.head(10))