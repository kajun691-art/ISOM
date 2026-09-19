import requests
from PIL import Image
from transformers import pipeline

def load_image_captioner():
    # Explicitly specify the recommended BLIP base model
    model_name = "Salesforce/blip-image-captioning-base"
    
    # Initialize the pipeline for the image-to-text task
    return pipeline("image-to-text", model=model_name)

def describe_image(caption_pipeline, image_path_or_url):
    # Fetch and open the image, converting it to RGB format
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")
        
    # Run the pipeline on the image
    result = caption_pipeline(image)
    
    # Extract the generated string from the pipeline output
    description = result[0]["generated_text"]
    return description

def main():
    print("Loading model (this may take a moment the first time)...")
    caption_pipeline = load_image_captioner()
    
    # A sample image of two parrots from Hugging Face's documentation dataset
    sample_image = "https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png"
    
    print("Generating description...")
    description = describe_image(caption_pipeline, sample_image)
    
    print("\n--- Output ---")
    print(f"Brief Description: {description}")

if __name__ == "__main__":
    main()
