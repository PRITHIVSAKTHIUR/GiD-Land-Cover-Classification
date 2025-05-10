import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/GiD-Land-Cover-Classification"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# ID to label mapping
id2label = {
    "0": "arbor woodland",
    "1": "artificial grassland",
    "2": "dry cropland",
    "3": "garden plot",
    "4": "industrial land",
    "5": "irrigated land",
    "6": "lake",
    "7": "natural grassland",
    "8": "paddy field",
    "9": "pond",
    "10": "river",
    "11": "rural residential",
    "12": "shrub land",
    "13": "traffic land",
    "14": "urban residential"
}

def detect_land_cover(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=detect_land_cover,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5, label="Land Cover Type"),
    title="GiD-Land-Cover-Classification",
    description="Upload an image to classify its land cover type: arbor woodland, dry cropland, lake, river, traffic land, etc."
)

if __name__ == "__main__":
    iface.launch()
