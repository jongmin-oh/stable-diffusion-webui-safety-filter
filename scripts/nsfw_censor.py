import os
from io import BytesIO
from PIL import Image

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForImageClassification, ViTImageProcessor

from modules import scripts

safety_model_id = "Falconsai/nsfw_image_detection"
model = None
processor = None

warning_image = os.path.join("extensions", "warning.png")


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
    
    warning_img = Image.open(warning_image).convert("RGB")
    
    for i, prob in enumerate(nsfw_probs):
        if prob > threshold:
            resized_warning = warning_img.resize(images[0].size)
            images[i] = resized_warning
    
    return images

class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW detect"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs['images']
        if args[0] is True:
            threshold = float(args[1])  # threshold를 float로 명시적 변환
            images[:] = nsfw_detect(images, threshold)

    def ui(self, is_img2img):
        enable_nsfw_detect = gr.Checkbox(label='Enable NSFW detect',
                                        value=False,
                                        elem_id=self.elem_id("enable_nsfw_detect"))
        threshold = gr.Slider(label="threshold",
                              minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                              elem_id=self.elem_id("threshold"))
        return [enable_nsfw_detect, threshold]
