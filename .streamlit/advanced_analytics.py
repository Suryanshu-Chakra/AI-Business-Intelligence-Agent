import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("data/sentiment_analysis_output.csv")

# Filter negative reviews
negative_reviews = df[df['Sentiment'] == 'NEGATIVE']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=20
)

X = vectorizer.fit_transform(
    negative_reviews['Cleaned_Reviews']
)

keywords = vectorizer.get_feature_names_out()

# Get scores
scores = X.sum(axis=0).A1

keyword_scores = list(zip(keywords, scores))

# Sort descending
keyword_scores = sorted(
    keyword_scores,
    key=lambda x: x[1],
    reverse=True
)

print("\n===== ADVANCED COMPLAINT ANALYSIS =====\n")

for word, score in keyword_scores:
    print(f"{word}: {round(score,2)}")