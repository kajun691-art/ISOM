"""
Magic Storybook Generator

A Streamlit web application designed for children aged 3-10. 
It processes an uploaded image, generates a descriptive caption, 
writes a magical bedtime story based on that caption, and reads it out loud.
"""

import io
import logging
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Configuration Constants
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
STORY_MODEL_NAME = "ykumards/smollm2-135m-bedtime-stories"


@st.cache_resource(show_spinner=False)
def load_caption_model():
    """
    Loads and caches the image captioning pipeline.
    Uses 'image-text-to-text' for compatibility with transformers v5+.
    """
    logger.info(f"Loading caption model: {CAPTION_MODEL_NAME}")
    return pipeline("image-text-to-text", model=CAPTION_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_story_model():
    """
    Loads and caches the text generation pipeline for story writing.
    Updated to use 'text-generation' for SmolLM-based models.
    """
    logger.info(f"Loading story model: {STORY_MODEL_NAME}")
    return pipeline("text-generation", model=STORY_MODEL_NAME)


def generate_image_caption(image: Image.Image, caption_pipeline) -> str:
    """
    Analyzes the uploaded image and returns a short text description.
    """
    try:
        # The 'image-text-to-text' pipeline requires a text prompt alongside the image.
        # We provide a generic starting phrase ("A picture of") for the AI to complete.
        result = caption_pipeline(images=image, text="A picture of", max_new_tokens=40)
        return result[0]["generated_text"].strip()
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        raise RuntimeError(f"Failed to generate a description for the image. Details: {str(e)}")


def generate_bedtime_story(caption: str, story_pipeline) -> str:
    """
    Expands an image caption into a child-friendly bedtime story.
    Adapted for a causal language model with decoding parameters to prevent looping.
    """
    # For a causal LM, we provide the beginning of the story as the prompt.
    prompt = f"Once upon a time in a magical land, {caption}. "
    
    try:
        output = story_pipeline(
            prompt, 
            max_new_tokens=100,          # Number of new words to generate
            do_sample=True, 
            temperature=0.7,             # Slightly focused temperature for better coherence
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,      # Penalizes the model for reusing words
            no_repeat_ngram_size=3,      # Prevents the model from repeating any 3-word phrase
            return_full_text=True        # Ensures the prompt is included in the output
        )
        return output[0]["generated_text"].strip()
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        raise RuntimeError(f"Failed to generate the story text. Details: {str(e)}")


def convert_text_to_audio(text: str) -> io.BytesIO:
    """
    Converts text to an MP3 audio stream using Google Text-to-Speech (gTTS).
    """
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        logger.error(f"Error synthesizing audio: {e}")
        raise RuntimeError("Failed to convert the story into audio.")


def setup_ui():
    """Configures the Streamlit page layout and textual headers."""
    st.set_page_config(
        page_title="Magic Storybook Generator",
        page_icon="📖",
        layout="centered"
    )
    st.title("✨ Magic Storybook Generator ✨")
    st.markdown("Upload a picture, and let the AI tell you a bedtime story!")


def main():
    """
    Main application orchestrator. Handles state, UI updates, and sequential 
    execution of the machine learning models.
    """
    setup_ui()

# 1. Load Models
    with st.spinner("Waking up the magic storytelling engine (this may take a moment)..."):
        try:
            caption_pipe = load_caption_model()
            story_pipe = load_story_model()
        except Exception as e:
            # We are adding {e} here to see the exact technical failure on the screen!
            st.error(f"Uh oh! The magic engine couldn't wake up. Details: {e}")
            logger.error(f"Model initialization failed: {e}")
            st.stop()

    
    # 2. File Upload Interface
    uploaded_file = st.file_uploader(
        "Choose a photo (PNG, JPG, JPEG):", 
        type=["png", "jpg", "jpeg"]
    )

    # 3. Process Uploaded File
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your Uploaded Picture", use_container_width=True)
        except Exception as e:
            st.error("We couldn't read that image file. Please try a different one!")
            logger.error(f"Image load error: {e}")
            st.stop()

        # 4. Trigger Story Generation Pipeline
        if st.button("🪄 Tell Me a Story!"):
            
            try:
                # Step A: Image to Text (Captioning)
                with st.spinner("Looking closely at your picture... 🔍"):
                    caption = generate_image_caption(image, caption_pipe)
                st.info(f"**What the AI sees:** {caption}")

                # Step B: Text to Text (Storytelling)
                with st.spinner("Spinning a magical tale... 🪄"):
                    story = generate_bedtime_story(caption, story_pipe)
                
                st.subheader("🌟 Your Story")
                st.write(story)
                
                word_count = len(story.split())
                st.caption(f"Story length: {word_count} words")

                # Step C: Text to Speech (Audio Synthesis)
                with st.spinner("Recording the storyteller's voice... 🎙️"):
                    audio_data = convert_text_to_audio(story)
                
                st.subheader("🔊 Listen to Your Story")
                st.audio(audio_data, format="audio/mp3")
                st.balloons()

            except RuntimeError as run_err:
                st.error(f"Oops! Something went wrong: {run_err}")
            except Exception as unk_err:
                st.error("An unexpected magical glitch occurred. Please try again!")
                logger.error(f"Unhandled pipeline error: {unk_err}")


if __name__ == "__main__":
    main()
