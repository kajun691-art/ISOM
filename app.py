import streamlit as st
from transformers import pipeline

# Cache the model loading so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    # Omitting a model selects the default model configured for sentiment analysis.
    return pipeline("sentiment-analysis")

st.title("Sentiment Analysis App")
st.write("Enter text below to determine if the sentiment is positive or negative.")

# Load the pipeline
sentiment_pipeline = load_model()

# Text input area with the default text from your notebook
default_text = "Deep Learning (DL) represents a highly promising approach to developing applications in Artificial Intelligence (AI)."
user_text = st.text_area("Input Text", value=default_text, height=150)

# Button to trigger analysis
if st.button("Analyze Sentiment"):
    if user_text.strip():
        # Perform sentiment analysis
        result = sentiment_pipeline(user_text)
        
        # Extract label and score
        label = result[0]["label"]
        score = result[0]["score"]
        
        # Display the results
        st.subheader("Result:")
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence Score:** {score:.4f}")
    else:
        st.warning("Please enter some text to analyze.")
