import pandas as pd
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
df = pd.read_csv("data/sentiment_analysis_output.csv")

# ==============================
# PRECOMPUTED BUSINESS METRICS
# ==============================

total_reviews = len(df)

positive_reviews = (df['Sentiment'] == 'POSITIVE').sum()
negative_reviews = (df['Sentiment'] == 'NEGATIVE').sum()

positive_percentage = round((positive_reviews / total_reviews) * 100, 2)
negative_percentage = round((negative_reviews / total_reviews) * 100, 2)

# Negative review analysis
negative_df = df[df['Sentiment'] == 'NEGATIVE']

all_text = " ".join(
    negative_df['Cleaned_Reviews'].astype(str)
)

words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())

filtered_words = [
    word for word in words
    if word not in ENGLISH_STOP_WORDS and len(word) > 2
]

word_counts = Counter(filtered_words)

top_keywords = word_counts.most_common(10)

# ==============================
# AI AGENT FUNCTION
# ==============================

def business_agent(query):

    query = query.lower()

    # Greeting
    if any(word in query for word in ["hi", "hello", "hey"]):
        return """
Hello! I am your AI Business Intelligence Agent.

You can ask me questions about:
- customer sentiment
- complaints
- battery issues
- screen problems
- product quality
- customer satisfaction
"""

    # Sentiment questions
    elif any(word in query for word in [
        "sentiment",
        "customer satisfaction",
        "positive",
        "negative",
        "happy",
        "unhappy"
    ]):

        return f"""
===== CUSTOMER SENTIMENT ANALYSIS =====

Total Reviews: {total_reviews}

Positive Reviews: {positive_percentage}%
Negative Reviews: {negative_percentage}%

Business Insight:
Customer dissatisfaction is relatively high.

Recommendation:
- Improve smartphone reliability
- Strengthen customer support
- Investigate recurring product defects
"""

    # Battery issues
    elif "battery" in query:

        return """
===== BATTERY ISSUE ANALYSIS =====

Battery-related complaints are frequently detected.

Possible Problems:
- fast battery drain
- low battery life
- charging issues

Recommended Actions:
- Improve battery optimization
- Enhance battery durability testing
- Investigate charging system quality
"""

    # Screen issues
    elif "screen" in query or "display" in query:

        return """
===== SCREEN ISSUE ANALYSIS =====

Customers frequently report screen/display problems.

Possible Problems:
- screen damage
- touch responsiveness
- display quality issues

Recommended Actions:
- Improve display testing
- Strengthen screen durability
- Reduce manufacturing defects
"""

    # Complaint analysis
    elif any(word in query for word in [
        "complaints",
        "issues",
        "problems",
        "customer problems"
    ]):

        keyword_text = "\n".join(
            [f"- {word}: {count}" for word, count in top_keywords]
        )

        return f"""
===== TOP CUSTOMER COMPLAINTS =====

Most Frequent Complaint Keywords:

{keyword_text}

Business Insight:
Recurring complaints indicate product quality concerns.

Recommendation:
Focus on:
- hardware reliability
- battery optimization
- screen quality
"""

    # Product quality
    elif any(word in query for word in [
        "quality",
        "performance",
        "reliability"
    ]):

        return """
===== PRODUCT QUALITY ANALYSIS =====

AI analysis suggests recurring product reliability concerns.

Common Areas:
- battery performance
- screen issues
- device durability
- customer usability concerns

Recommended Actions:
- Improve hardware testing
- Increase QA validation
- Improve long-term device durability
"""

    # Invalid questions
    else:

        return """
I can only answer business intelligence questions related to smartphone customer reviews.

Please ask about:
- sentiment analysis
- customer complaints
- battery issues
- screen problems
- product quality
- customer satisfaction
"""

# ==============================
# CHAT LOOP
# ==============================

print("\nAI Business Intelligence Agent Ready!")

while True:

    user_query = input("\nAsk a business question (or type 'exit'): ")

    if user_query.lower() == "exit":
        print("\nExiting AI Agent...")
        break

    response = business_agent(user_query)

    print("\n" + response)