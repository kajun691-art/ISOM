import io
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# Configure Streamlit page for kids
st.set_page_config(
    page_title="Magic Storybook Generator",
    page_icon="📖",
    layout="centered"
)

st.title("✨ Magic Storybook Generator ✨")
st.markdown("Upload a picture, and let the AI tell you a bedtime story!")

# -------------------------------------------------------------
# Model Loading (Cached to run once on startup)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_image_captioner():
    """Loads the BLIP image-to-text pipeline."""
    return pipeline(
        "image-to-text", 
        model="Salesforce/blip-image-captioning-base"
    )

@st.cache_resource(show_spinner=False)
def load_story_generator():
    """Loads a lightweight sequence-to-sequence model for prompt-following."""
    return pipeline(
        "text2text-generation", 
        model="google/flan-t5-base"
    )

with st.spinner("Waking up the magic storytelling engine..."):
    caption_pipe = load_image_captioner()
    story_pipe = load_story_generator()

# -------------------------------------------------------------
# Core Helper Functions
# -------------------------------------------------------------
def generate_caption(image: Image.Image) -> str:
    """Extracts a short descriptive caption from an image."""
    result = caption_pipe(image, max_new_tokens=40)
    return result[0]["generated_text"].strip()

def generate_story(caption: str) -> str:
    """Expands the image caption into a child-friendly short story (50-100 words)."""
    prompt = (
        f"Write a cheerful, magical, imaginative bedtime story for a 6-year-old child "
        f"in 60 to 90 words based on this scene: '{caption}'. "
        f"Use friendly, simple words and a happy ending."
    )
    
    output = story_pipe(
        prompt, 
        max_length=150, 
        min_length=60, 
        do_sample=True, 
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
    return output[0]["generated_text"].strip()

def text_to_speech(text: str) -> io.BytesIO:
    """Converts the story text to speech using gTTS and returns an in-memory MP3 buffer."""
    tts = gTTS(text=text, lang="en", slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# -------------------------------------------------------------
# User Interface
# -------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a photo (PNG, JPG, JPEG):", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Uploaded Picture", use_container_width=True)

    if st.button("🪄 Tell Me a Story!"):
        # Step 1: Captioning
        with st.spinner("Looking closely at your picture... 🔍"):
            caption = generate_caption(image)
        st.info(f"**What the AI sees:** {caption}")

        # Step 2: Story Generation
        with st.spinner("Spinning a magical tale... 🪄"):
            story = generate_story(caption)

        st.subheader("🌟 Your Story")
        st.write(story)

        # Word count check
        word_count = len(story.split())
        st.caption(f"Story length: {word_count} words")

        # Step 3: Text to Speech
        with st.spinner("Recording the storyteller's voice... 🎙️"):
            audio_data = text_to_speech(story)

        st.subheader("🔊 Listen to Your Story")
        st.audio(audio_data, format="audio/mp3")
