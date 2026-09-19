from transformers import pipeline

pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
pipe("middleagedMan.jpg")
