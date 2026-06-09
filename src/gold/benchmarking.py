from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
# testing the model

text1 = "I absolutely love using HuggingFace models for NLP tasks!"

result = sentiment_analyzer(text1)
print(result)

