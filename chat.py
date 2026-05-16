import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load analyzed dataset
df = pd.read_csv("data/sentiment_analysis_output.csv")

# Use only useful columns
reviews = df['Reviews'].astype(str).tolist()

# Load embedding model
print("Loading AI chatbot model...")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
review_embeddings = model.encode(
    reviews,
    convert_to_tensor=True
)

print("Chatbot ready!")

# Business recommendation function
def generate_business_response(query, matched_review):

    query_lower = query.lower()

    if "unhappy" in query_lower or "negative" in query_lower:
        return f"""
Customers appear dissatisfied mainly due to product quality and usability concerns.

Example customer feedback:
"{matched_review}"

Recommended Actions:
- Improve product quality testing
- Strengthen battery and screen durability
- Enhance customer support response
"""

    elif "battery" in query_lower:
        return f"""
Battery-related complaints detected.

Example review:
"{matched_review}"

Recommendation:
- Improve battery optimization
- Increase battery life testing
"""

    elif "screen" in query_lower:
        return f"""
Screen/display issues are frequently mentioned.

Example review:
"{matched_review}"

Recommendation:
- Improve display quality assurance
- Reduce screen defect rates
"""

    else:
        return f"""
Based on customer feedback analysis:

Relevant Review:
"{matched_review}"

AI Insight:
Customer sentiment suggests improvement opportunities in smartphone performance and reliability.
"""

# Chat loop
while True:

    query = input("\nAsk a business question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # Encode query
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    # Find most similar review
    similarity_scores = util.cos_sim(
        query_embedding,
        review_embeddings
    )[0]

    best_match_index = similarity_scores.argmax()

    matched_review = reviews[best_match_index]

    # Generate response
    response = generate_business_response(
        query,
        matched_review
    )

    print("\n===== AI BUSINESS CHATBOT =====")
    print(response)