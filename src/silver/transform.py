import src.bronze.extract as e 

# removing unclear comments
# first checking how many they are

df_raw = e.df_raw
sentiment_mapping = e.sentiment_mapping 

counting = df_raw["example_very_unclear"].value_counts()

print(counting)

# inspecint the type of values in example_very_unclear column 

print(df_raw["example_very_unclear"].unique())
print(df_raw["example_very_unclear"].dtype)

mask = df_raw['example_very_unclear'] == False

filtered_df = df_raw[mask]

aggregate_data = filtered_df.groupby(['id', 'text']).agg(
    admiration=('admiration', 'sum'),
    amusement=('amusement', 'sum'),
    anger=('anger', 'sum'),
    annoyance=('annoyance', 'sum'),
    approval=('approval', 'sum'),
    caring=('caring', 'sum'),
    confusion=('confusion', 'sum'),
    curiosity=('curiosity', 'sum'),
    desire=('desire', 'sum'),
    disappointment=('disappointment', 'sum'),
    disapproval=('disapproval', 'sum'),
    disgust=('disgust', 'sum'),
    embarrassment=('embarrassment', 'sum'),
    excitement=('excitement', 'sum'),
    fear=('fear', 'sum'),
    gratitude=('gratitude', 'sum'),
    grief=('grief', 'sum'),
    joy=('joy', 'sum'),
    love=('love', 'sum'),
    neurvousness=('nervousness', 'sum'),
    optimism=('optimism', 'sum'),
    pride=('pride', 'sum'),
    realization=('realization', 'sum'),
    relief=('relief', 'sum'),
    remorse=('remorse', 'sum'),
    sadness=('sadness', 'sum'),
    surprise=('surprise', 'sum'),
    neutral=('neutral', 'sum')
)

# mapping the json sentiment dict
emotion_to_sentiment = {}

for sentiment, emotions in sentiment_mapping.items():
    for emotion in emotions:
        if sentiment == "ambiguous":
            emotion_to_sentiment[emotion] = "neutral"
        else:
            emotion_to_sentiment[emotion] = sentiment 

# in the GoEmotions dataset, there's a voting option for "neutral", but "neutral" as an emotion is NOT stated in the sentiment_mapping. Therefore, this will add neutral emotion for neutral sentiment, so values in the aggregated table are not nulls and the later comparison with RoBERTa also evaluate comments evaluated as neutrals (emotion).
emotion_to_sentiment['neutral'] = 'neutral'

# finding maximum value from rows

aggregate_data['Highest Score'] = aggregate_data.idxmax(axis=1)

# append column from emotion_to_sentiment mapping to aggregate_data

aggregate_data["sentiment"] = aggregate_data['Highest Score'].map(emotion_to_sentiment)

# saving the csv transformed dataset

# aggregate_data.to_csv('transformed_goemotions_dataset.csv')