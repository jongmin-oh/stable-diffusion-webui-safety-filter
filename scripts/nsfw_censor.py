import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, ViTImageProcessor


falcon_model = None
falcon_processor = None
vit_model = None
vit_processor = None


def nsfw_detect(images, nsfw_detect_model):
    if nsfw_detect_model == "vit":
        return nsfw_vit_detect(images)
    else:
        return nsfw_falcon_detect(images)

def nsfw_falcon_detect(images):
    global falcon_model, falcon_processor
    
    if falcon_model is None:
        falcon_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
        falcon_processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
    
    with torch.no_grad():
        inputs = falcon_processor(images=images, return_tensors="pt")
        outputs = falcon_model(**inputs)
        logits = outputs.logits
    
    probabilities = F.softmax(logits, dim=1)
    nsfw_probs = probabilities[:, 1].tolist()
    
    return images, nsfw_probs[0]


def nsfw_vit_detect(images):
    global vit_model, vit_processor
    
    if vit_model is None:
        vit_model = AutoModelForImageClassification.from_pretrained('AdamCodd/vit-base-nsfw-detector')
        vit_processor = ViTImageProcessor.from_pretrained('AdamCodd/vit-base-nsfw-detector')
    
    with torch.no_grad():
        inputs = vit_processor(images=images, return_tensors="pt")
        outputs = vit_model(**inputs)
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=1)
    nsfw_probs = probabilities[:, 1].tolist()
    
    return images, nsfw_probs[0]
