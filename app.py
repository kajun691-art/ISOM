import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    # Explicitly specifying the default model from the notebook
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    return pipeline("sentiment-analysis", model=model_name)

def analyze_text(sentiment_pipeline, text):
    # Processes the text and extracts the label and score
    result = sentiment_pipeline(text)
    label = result[0]["label"]
    score = result[0]["score"]
    return label, score

def main():
    st.title("Sentiment Analysis App")
    st.write("Enter text below to determine if the sentiment is positive or negative.")

    # Load the pipeline
    sentiment_pipeline = load_model()

    # Text input area with default text
    default_text = "Deep Learning (DL) represents a highly promising approach to developing applications in Artificial Intelligence (AI)."
    user_text = st.text_area("Input Text", value=default_text, height=150)

    # Button to trigger analysis
    if st.button("Analyze Sentiment"):
        if user_text.strip():
            # Call the helper function
            label, score = analyze_text(sentiment_pipeline, user_text)
            
            # Display the results
            st.subheader("Result:")
            st.write(f"**Sentiment:** {label}")
            st.write(f"**Confidence Score:** {score:.4f}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
