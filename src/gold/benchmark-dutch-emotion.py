import pandas as pd
from src.gold.benchmarking import analyze_sentiment

df_emotioNL = pd.read_csv("data/emotioNL_sample.csv")

texts = df_emotioNL[["ID", "Text"]].rename(
    columns={
        "ID": "id",
        "Text": "text"
    }
).to_dict("records")

emotion_to_sentiment = {
    "joy": "positive",
    "love": "positive",
    "anger": "negative",
    "sadness": "negative",
    "fear": "negative",
    "neutral": "neutral"
}

df_emotioNL["expected_sentiment"] = df_emotioNL["Category"].map(emotion_to_sentiment)

predictions = analyze_sentiment(texts)

df_emotioNL_predictions = pd.DataFrame(predictions)

df_compare = pd.merge(
    df_emotioNL,
    df_emotioNL_predictions,
    left_on="ID",
    right_on="id"
)

df_compare["prediction_match"] = (
    df_compare["expected_sentiment"] == df_compare["roberta_prediction"]
)

accuracy = df_compare["prediction_match"].mean()

print(f"Accuracy: {accuracy:.2%}")

print(
    df_compare[
        [
            "ID",
            "Text",
            "Category",
            "expected_sentiment",
            "roberta_prediction",
            "prediction_match"
        ]
    ].head(10)
)

print(df_compare["expected_sentiment"].value_counts())
print(df_compare["roberta_prediction"].value_counts())