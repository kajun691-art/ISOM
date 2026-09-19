import requests
from PIL import Image
from transformers import pipeline

def load_captioning_model():
    # Loading the model once and reusing it is key for efficiency.
    # We use BLIP base because it is lightweight, fast, and yields concise text.
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def get_image_description(model_pipeline, image_source):
    # Handle both web URLs and local file paths
    if image_source.startswith("http://") or image_source.startswith("https://"):
        image = Image.open(requests.get(image_source, stream=True).raw)
    else:
        image = Image.open(image_source)
        
    # Convert to RGB to ensure compatibility with the model
    image = image.convert("RGB")
    
    # Generate the brief description
    output = model_pipeline(image)
    
    # Extract the generated text from the pipeline's output list
    return output[0]["generated_text"]

if __name__ == "__main__":
    # 1. Initialize the pipeline
    print("Loading model...")
    captioner = load_captioning_model()
    
    # 2. Provide an image URL (or local path)
    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
    
    # 3. Generate and print the description
    print("Analyzing image...")
    description = get_image_description(captioner, sample_image_url)
    print(f"Brief Description: {description}")
