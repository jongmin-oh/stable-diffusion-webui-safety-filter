import os
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, ViTImageProcessor


safety_model_id = "Falconsai/nsfw_image_detection"
model = None
processor = None


def nsfw_detect(images, threshold=0.5):
    global model, processor
    
    if model is None:
        model = AutoModelForImageClassification.from_pretrained(safety_model_id)
        processor = ViTImageProcessor.from_pretrained(safety_model_id)
    
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
    
    probabilities = F.softmax(logits, dim=1)
    nsfw_probs = probabilities[:, 1].tolist()
    
    for i, prob in enumerate(nsfw_probs):
        if prob > threshold:
            black_image = Image.new('RGB', images[0].size, color='black')
            images[i] = black_image
    
    return images