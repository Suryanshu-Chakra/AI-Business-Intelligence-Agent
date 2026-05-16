import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

/* Main background */
.main {
    background-color: #0E1117;
}

/* KPI Cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1E1E1E, #262730);
    border: 1px solid #00C2FF;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,194,255,0.2);
}

/* Chat messages */
.stChatMessage {
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161A23;
}

/* Headers */
h1, h2, h3 {
    color: #00C2FF;
}

/* Buttons */
.stButton>button {
    background-color: #00C2FF;
    color: white;
    border-radius: 10px;
    border: none;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Business Intelligence Agent",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("data/sentiment_analysis_output.csv")

# =====================================
# BUSINESS METRICS
# =====================================

total_reviews = len(df)

positive_reviews = (df['Sentiment'] == 'POSITIVE').sum()
negative_reviews = (df['Sentiment'] == 'NEGATIVE').sum()

positive_percentage = round((positive_reviews / total_reviews) * 100, 2)
negative_percentage = round((negative_reviews / total_reviews) * 100, 2)

# Complaint analysis
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

# =====================================
# AI AGENT FUNCTION
# =====================================

def business_agent(query):

    query = query.lower()

    if any(word in query for word in ["hi", "hello", "hey"]):
        return """
Hello! I am your AI Business Intelligence Agent.

You can ask me about:
• customer sentiment
• complaints
• battery issues
• screen problems
• product quality
• customer satisfaction
"""

    elif any(word in query for word in [
        "sentiment",
        "customer satisfaction",
        "positive",
        "negative",
        "happy",
        "unhappy"
    ]):

        return f"""
📊 CUSTOMER SENTIMENT ANALYSIS

• Total Reviews: {total_reviews}
• Positive Reviews: {positive_percentage}%
• Negative Reviews: {negative_percentage}%

Business Insight:
Customer dissatisfaction is relatively high.

Recommended Actions:
• Improve smartphone reliability
• Strengthen customer support
• Investigate recurring defects
"""

    elif "battery" in query:

        return """
🔋 BATTERY ISSUE ANALYSIS

Common Problems:
• Fast battery drain
• Charging issues
• Low battery life

Recommendations:
• Improve battery optimization
• Enhance durability testing
• Improve charging reliability
"""

    elif "screen" in query or "display" in query:

        return """
📱 SCREEN ISSUE ANALYSIS

Detected Problems:
• Display defects
• Touch responsiveness issues
• Screen durability concerns

Recommendations:
• Improve display QA testing
• Reduce manufacturing defects
• Strengthen screen durability
"""

    elif any(word in query for word in [
        "complaints",
        "issues",
        "problems"
    ]):

        keyword_text = "\n".join(
            [f"• {word}: {count}" for word, count in top_keywords]
        )

        return f"""
⚠ TOP CUSTOMER COMPLAINTS

Most Frequent Keywords:

{keyword_text}

Business Insight:
Recurring issues suggest quality concerns.

Recommendations:
• Improve hardware reliability
• Enhance battery optimization
• Improve screen quality
"""

    elif any(word in query for word in [
        "quality",
        "performance",
        "reliability"
    ]):

        return """
🛠 PRODUCT QUALITY ANALYSIS

Key Concerns:
• Battery performance
• Screen reliability
• Device durability
• Customer usability issues

Recommendations:
• Improve hardware testing
• Increase QA validation
• Improve long-term durability
"""

    else:

        return """
I can only answer smartphone business intelligence questions.

Try asking about:
• customer sentiment
• complaints
• battery issues
• screen problems
• product quality
"""

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🤖 AI Business Agent")

st.sidebar.info("""
AI-powered smartphone review intelligence system using:
- HuggingFace Transformers
- NLP Analytics
- Business Intelligence
- Sentiment Analysis
""")

# =====================================
# MAIN TITLE
# =====================================

st.title("📊 AI-Powered Smartphone Review Intelligence System")

# =====================================
# KPI SECTION
# =====================================

col1, col2, col3 = st.columns(3)

col1.metric("Total Reviews", total_reviews)
col2.metric("Positive Reviews %", f"{positive_percentage}%")
col3.metric("Negative Reviews %", f"{negative_percentage}%")

# =====================================
# CHARTS
# =====================================

left_col, right_col = st.columns(2)

# Pie chart
with left_col:

    st.subheader("📈 Sentiment Distribution")

    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    fig = px.pie(
        sentiment_counts,
        values='Count',
        names='Sentiment'
    )

    st.plotly_chart(fig, width='stretch')

# Complaint keywords
with right_col:

    st.subheader("⚠ Complaint Keywords")

    keyword_df = pd.DataFrame(
        top_keywords,
        columns=['Keyword', 'Frequency']
    )

    fig2 = px.bar(
        keyword_df,
        x='Keyword',
        y='Frequency'
    )

    st.plotly_chart(fig2, width='stretch')

# =====================================
# CHATBOT SECTION
# =====================================

st.subheader("💬 AI Business Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input(
    "Ask a business intelligence question..."
)

if user_input:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    ai_response = business_agent(user_input)

    # Store AI response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)

# =====================================
# REVIEW EXPLORER
# =====================================

st.subheader("📝 Customer Review Explorer")

sentiment_filter = st.selectbox(
    "Filter by Sentiment",
    ['ALL', 'POSITIVE', 'NEGATIVE']
)

if sentiment_filter != 'ALL':
    filtered_df = df[df['Sentiment'] == sentiment_filter]
else:
    filtered_df = df

st.dataframe(
    filtered_df[
        ['Product Name', 'Rating', 'Sentiment', 'Confidence_Score']
    ].head(50),
    width='stretch'
)

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "AI-Powered Business Intelligence Agent using HuggingFace Transformers"
)