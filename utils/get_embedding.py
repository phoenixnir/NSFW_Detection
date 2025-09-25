import numpy as np
import torch
from PIL import Image

def get_image_embedding(image_path, model, processor):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

def get_text_embedding(text, model, processor):
    inputs = processor(text=text, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    input_ids = inputs['input_ids']
    input_ids = input_ids.argmax(dim=-1)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features, input_ids