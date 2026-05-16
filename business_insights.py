import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Load sentiment output
df = pd.read_csv("data/sentiment_analysis_output.csv")

# Filter negative reviews
negative_reviews = df[df['Sentiment'] == 'NEGATIVE']

# Combine all negative review text
all_text = " ".join(negative_reviews['Cleaned_Reviews'].astype(str))

# Extract words
words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())

# Remove stopwords
filtered_words = [
    word for word in words
    if word not in ENGLISH_STOP_WORDS and len(word) > 2
]

# Count keyword frequency
word_counts = Counter(filtered_words)

# Top complaint keywords
top_keywords = word_counts.most_common(15)

print("\n===== TOP CUSTOMER COMPLAINT KEYWORDS =====\n")

for word, count in top_keywords:
    print(f"{word}: {count}")

# KPI Calculations
total_reviews = len(df)
positive_reviews = (df['Sentiment'] == 'POSITIVE').sum()
negative_reviews_count = (df['Sentiment'] == 'NEGATIVE').sum()

positive_percentage = round((positive_reviews / total_reviews) * 100, 2)
negative_percentage = round((negative_reviews_count / total_reviews) * 100, 2)

print("\n===== BUSINESS KPI SUMMARY =====\n")

print(f"Total Reviews: {total_reviews}")
print(f"Positive Review %: {positive_percentage}%")
print(f"Negative Review %: {negative_percentage}%")

# Business Recommendations
print("\n===== AI BUSINESS RECOMMENDATIONS =====\n")

if negative_percentage > 50:
    print("- Customer dissatisfaction is high.")
    print("- Company should investigate recurring product issues.")
    print("- Focus on customer support and product quality improvement.")

if 'battery' in [word for word, count in top_keywords]:
    print("- Battery-related complaints are frequent.")
    print("- Improve battery optimization and durability.")

if 'screen' in [word for word, count in top_keywords]:
    print("- Screen/display issues detected.")
    print("- Review display quality testing process.")

if 'phone' in [word for word, count in top_keywords]:
    print("- Core product experience complaints detected.")
    print("- Product reliability needs improvement.")