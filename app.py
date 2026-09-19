import streamlit as st
import requests

# Hugging Face API URL for the BLIP model
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

# Fetch the API token securely from Streamlit's secrets
headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

def describe_image(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.json()

st.title("Image Captioning App")
st.write("Upload an image to get a brief description.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")
    
    if st.button("Generate Description"):
        with st.spinner("Analyzing image..."):
            # Read the file directly as bytes
            image_bytes = uploaded_file.getvalue()
            
            # Send to Hugging Face API
            result = describe_image(image_bytes)
            
            # Verify if the API returned the expected result
            if isinstance(result, list) and "generated_text" in result[0]:
                st.success(f"**Description:** {result[0]['generated_text']}")
            else:
                # This will catch API rate limits or invalid tokens
                st.error(f"API Error: {result}")
