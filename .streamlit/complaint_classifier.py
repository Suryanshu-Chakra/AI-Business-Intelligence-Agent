import pandas as pd

# Load dataset
df = pd.read_csv("data/sentiment_analysis_output.csv")

# Complaint categories
complaint_categories = {
    "Battery Issues": [
        "battery",
        "charging",
        "charge",
        "drain"
    ],

    "Screen Issues": [
        "screen",
        "display",
        "touch"
    ],

    "Performance Issues": [
        "slow",
        "lag",
        "freeze",
        "hang"
    ],

    "Camera Issues": [
        "camera",
        "photo",
        "picture"
    ]
}

# Initialize counters
category_counts = {
    category: 0
    for category in complaint_categories
}

# Analyze negative reviews
negative_reviews = df[
    df['Sentiment'] == 'NEGATIVE'
]

for review in negative_reviews['Cleaned_Reviews']:

    review = str(review).lower()

    for category, keywords in complaint_categories.items():

        if any(keyword in review for keyword in keywords):
            category_counts[category] += 1

print("\n===== COMPLAINT CATEGORY ANALYSIS =====\n")

for category, count in category_counts.items():
    print(f"{category}: {count}")