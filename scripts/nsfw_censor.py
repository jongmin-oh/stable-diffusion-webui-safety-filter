import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, ViTImageProcessor


safety_model_id = "Falconsai/nsfw_image_detection"
model = None
processor = None


def nsfw_detect(images):
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
    
    return images, nsfw_probs[0]
