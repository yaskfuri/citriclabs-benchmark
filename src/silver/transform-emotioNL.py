import pandas as pd
from src.utils.sampling import cochran_sample

df = pd.read_csv(
    "data/raw/corpus_fulltext_captions.txt",
    sep="\t"
)

df_sample = cochran_sample(df)

df_sample.to_csv(
    "data/emotioNL_sample.csv",
    index=False
)