from transformers import pipeline
from PIL import Image

# Streamlit UI
print("Title: Age Classification using ViT")

# Load the age classification pipeline
# The code below should be placed in the main part of the program
age_classifier = pipeline("image-classification",
                          model="nateraw/vit-age-classifier")

image_name = "middleagedMan.jpg"
image_name = Image.open(image_name).convert("RGB")

# Classify age
age_predictions = age_classifier(image_name)
print(age_predictions)
age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

# Display results
print("Predicted Age Range:")
print(f"Age range: {age_predictions[0]['label']}")
