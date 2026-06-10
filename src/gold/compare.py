import pandas as pd

df_roberta = pd.read_csv('data/goemotions_roberta_prediction.csv')

df_sample = pd.read_csv('data/goemotions_sample.csv')

df_merge = pd.merge(df_sample, df_roberta, on="id")

df_merge['prediction_match'] = df_merge['roberta_prediction'] == df_merge["sentiment"]

print(df_merge[["id", "sentiment", "roberta_prediction", "prediction_match"]].head(10))

accuracy = df_merge["prediction_match"].mean()
acc = accuracy * 100
round(acc, 2)
print(round(acc, 2))