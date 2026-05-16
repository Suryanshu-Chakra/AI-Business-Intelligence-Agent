# =========================================
# AI TEXT SUMMARIZATION TOOL
# =========================================

import streamlit as st
from transformers import pipeline

# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource
def load_model():

    summarizer = pipeline(
        task="summarization",
        model="facebook/bart-large-cnn"
    )

    return summarizer

summarizer = load_model()

# =========================================
# UI
# =========================================

st.set_page_config(
    page_title="AI Summarization Tool",
    layout="wide"
)

st.title("🤖 AI-Based Text Summarization Tool")

st.write(
    "Enter long text and generate AI-powered summaries using Transformers."
)

# =========================================
# INPUT
# =========================================

input_text = st.text_area(
    "📄 Enter Your Text",
    height=300
)

# =========================================
# SUMMARIZATION
# =========================================

if st.button("Generate Summary"):

    if input_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        with st.spinner("Generating summary..."):

            summary = summarizer(
                input_text,
                max_length=130,
                min_length=30,
                do_sample=False
            )

        # =========================================
        # OUTPUT
        # =========================================

        st.subheader("📝 AI Summary")

        st.success(summary[0]['summary_text'])

        # =========================================
        # STATS
        # =========================================

        st.subheader("📊 Text Statistics")

        original_words = len(input_text.split())
        summary_words = len(summary[0]['summary_text'].split())

        col1, col2 = st.columns(2)

        col1.metric("Original Words", original_words)
        col2.metric("Summary Words", summary_words)

        reduction = round(
            (1 - summary_words/original_words) * 100,
            2
        )

        st.metric("Reduction %", f"{reduction}%")