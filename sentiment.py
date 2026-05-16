import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load cleaned dataset
df = pd.read_csv("data/cleaned_mobile_reviews.csv")

# Keep only necessary rows
df = df.dropna(subset=['Cleaned_Reviews'])

# OPTIONAL:
# Use smaller sample initially for faster testing
df = df.head(1000)

# Load HuggingFace sentiment pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Lists to store outputs
sentiments = []
scores = []

print("Running sentiment analysis...")

# Analyze reviews
for review in tqdm(df['Cleaned_Reviews']):
    try:
        result = sentiment_pipeline(review[:512])[0]

        sentiments.append(result['label'])
        scores.append(result['score'])

    except Exception as e:
        sentiments.append("ERROR")
        scores.append(0)

# Add results to dataframe
df['Sentiment'] = sentiments
df['Confidence_Score'] = scores

# Save analyzed dataset
df.to_csv(
    "data/sentiment_analysis_output.csv",
    index=False,
    encoding="utf-8"
)

# KPI Summary
positive_count = (df['Sentiment'] == 'POSITIVE').sum()
negative_count = (df['Sentiment'] == 'NEGATIVE').sum()

print("\n===== SENTIMENT ANALYSIS SUMMARY =====")
print(f"Total Reviews Analyzed: {len(df)}")
print(f"Positive Reviews: {positive_count}")
print(f"Negative Reviews: {negative_count}")

print("\nTop 5 Results:")
print(df[['Cleaned_Reviews', 'Sentiment', 'Confidence_Score']].head())