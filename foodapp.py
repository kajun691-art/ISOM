import gradio as gr
from transformers import pipeline

# Initialize the image segmentation pipeline
# Note: device=0 can be used if you are running on a GPU, otherwise set to -1 for CPU
pipe = pipeline("image-segmentation", model="sayeed99/segformer_b3_clothes")

def segment_clothing(image):
    """
    Takes an input PIL Image and returns the segmentation overlay/mask output.
    """
    if image is None:
        return None
    
    # Run the pipeline
    results = pipe(image)
    
    # The pipeline returns a list of dictionaries with 'score', 'label', and 'mask'
    # Gradio's ImageSegmentation component accepts a tuple of (image, [list of dictionaries])
    return (image, results)

# Create the Gradio interface
demo = gr.Interface(
    fn=segment_clothing,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.ImageSegmentation(label="Segmented Clothing & Parts"),
    title="Clothing & Human Parsing Segmentation",
    description="Upload an image to segment clothing items, accessories, and body parts using the sayeed99/segformer_b3_clothes model.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
