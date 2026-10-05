import gradio as gr
from transformers import pipeline

# Initialize the image segmentation pipeline
pipe = pipeline("image-segmentation", model="sayeed99/segformer_b3_clothes")

def segment_clothing(image):
    """
    Takes an input PIL Image and returns the original image layered with segmentation masks.
    """
    if image is None:
        return None
    
    # Run the pipeline
    results = pipe(image)
    
    # The pipeline returns a list of dicts: [{'score': float, 'label': str, 'mask': PIL.Image}]
    # gr.AnnotatedImage expects a tuple: (original_image, [(mask, label), ...])
    annotations = []
    for result in results:
        annotations.append((result['mask'], result['label']))
        
    return (image, annotations)

# Create the Gradio interface
demo = gr.Interface(
    fn=segment_clothing,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.AnnotatedImage(label="Segmented Clothing & Parts"),
    title="Clothing & Human Parsing Segmentation",
    description="Upload an image to segment clothing items, accessories, and body parts using the sayeed99/segformer_b3_clothes model."
    # Removed allow_flagging="never" to maintain compatibility with Gradio 4+
)

if __name__ == "__main__":
    demo.launch()
