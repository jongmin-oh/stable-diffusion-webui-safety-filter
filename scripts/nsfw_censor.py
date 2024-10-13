import os
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoModelForImageClassification, ViTImageProcessor

from modules import scripts

safety_model_id = "Falconsai/nsfw_image_detection"

model = AutoModelForImageClassification.from_pretrained(safety_model_id)
processor = ViTImageProcessor.from_pretrained(safety_model_id)

warning_image = os.path.join("extensions", "warning.png")

def nsfw_detect(images, threshold=0.5):
    pil_images = [Image.fromarray((img.permute(1, 2, 0) * 255).byte().numpy()) for img in images]
    
    with torch.no_grad():
        inputs = processor(images=pil_images, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
    
    probabilities = F.softmax(logits, dim=1)
    nsfw_probs = probabilities[:, 1].tolist()
    
    warning_img = Image.open(warning_image).convert("RGB")
    
    for i, prob in enumerate(nsfw_probs):
        if prob > threshold:
            resized_warning = warning_img.resize((images[i].shape[2], images[i].shape[1]))
            images[i] = torch.from_numpy(np.array(resized_warning)).permute(2, 0, 1).float() / 255.0
    
    return images


def apply_nsfw_filter(images, threshold):
    tensor_images = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1).float() for img in images])
    filtered_images = nsfw_detect(tensor_images, threshold)
    return [Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8)) for img in filtered_images]

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
