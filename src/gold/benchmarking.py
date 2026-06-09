# from transformers import pipeline
from src.silver import transform as s 
import math

# sentiment analysis RoBERTa model used by CitricLabs

#sentiment_analyzer = pipeline(
#    "sentiment-analysis",
#    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
#)

# sampling the dataset using cochran's formula

CONFIDENCE_Z = 0.95
P = 0.5
ERROR = 0.05
N = len(s.aggregate_data)

n0 = (CONFIDENCE_Z**2 * P * (1 - P)) / (ERROR**2)

# Finite population size
n = n0 / (1 + ((n0 - 1) / N))

sample_size = math.ceil(n)

print(f"Sample size is: {sample_size}")

df_sample = s.aggregate_data.sample(n=sample_size)

# 




