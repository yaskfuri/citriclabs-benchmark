from src.silver import transform as s 
from transformers import pipeline 
import pandas as pd
import math

# sentiment analysis RoBERTa model used by CitricLabs

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)

# sampling the dataset using cochran's formula

CONFIDENCE_Z = 1.96
P = 0.5
ERROR = 0.05
N = len(s.aggregate_data)

n0 = (CONFIDENCE_Z**2 * P * (1 - P)) / (ERROR**2)

# Finite population size
n = n0 / (1 + ((n0 - 1) / N))

sample_size = math.ceil(n)

print(f"Sample size is: {sample_size}")

df_sample = s.aggregate_data.sample(n=sample_size)

# transforming text column into text list for the loop

df_sample = df_sample.reset_index()

# texts = df_sample['text'].tolist()

# function to analyze sentiment
sentiment_analysis = {} 

def analyze_sentiment(texts):
    only_texts = [row["text"] for row in texts]

    results = sentiment_analyzer(only_texts, top_k=None)

    sentiments = []

    for row, res in zip(texts, results):
        scores = {
            item["label"]: item["score"]
            for item in res
        }

        roberta_prediction = max(scores, key=scores.get)

        sentiments.append({
            "id": row["id"],
            "negative_score": round(scores["negative"] * 100, 2),
            "neutral_score": round(scores["neutral"] * 100, 2),
            "positive_score": round(scores["positive"] * 100, 2),
            "roberta_prediction": roberta_prediction
        })

    return sentiments

texts = df_sample[["id", "text"]].to_dict("records")

predictions = analyze_sentiment(texts)

df_predict = pd.DataFrame(predictions)

# df_predict.to_csv("data/goemotions_roberta_prediction.csv")

df_merge = pd.merge(df_sample, df_predict, on="id")
