![GiD.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Ux6BK8vxbJ1HBDjChDmfv.png)

# **GiD-Land-Cover-Classification**

> **GiD-Land-Cover-Classification** is a multi-class image classification model based on `google/siglip2-base-patch16-224`, trained to detect **land cover types** in geographical or environmental imagery. This model can be used for **urban planning**, **agriculture monitoring**, and **environmental analysis**.

```py
Classification Report:
                      precision    recall  f1-score   support

      arbor woodland     0.8868    0.9130    0.8997      2000
artificial grassland     0.9173    0.9425    0.9297      2000
        dry cropland     0.9320    0.9395    0.9358      2000
         garden plot     0.8639    0.8380    0.8508      2000
     industrial land     0.8967    0.8940    0.8953      2000
      irrigated land     0.8817    0.7865    0.8314      2000
                lake     0.7597    0.8045    0.7814      2000
   natural grassland     0.9770    0.9750    0.9760      2000
         paddy field     0.9305    0.9580    0.9441      2000
                pond     0.7646    0.7405    0.7523      2000
               river     0.8124    0.7945    0.8033      2000
   rural residential     0.8875    0.8325    0.8591      2000
          shrub land     0.8936    0.9195    0.9064      2000
        traffic land     0.9577    0.9510    0.9543      2000
   urban residential     0.7821    0.8470    0.8133      2000

            accuracy                         0.8757     30000
           macro avg     0.8762    0.8757    0.8755     30000
        weighted avg     0.8762    0.8757    0.8755     30000
```

![Untitled.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/09-cU54xSrM97DKD66LeU.png)

---

## **Label Classes**

The model distinguishes between the following land cover types:

```
0: arbor woodland  
1: artificial grassland  
2: dry cropland  
3: garden plot  
4: industrial land  
5: irrigated land  
6: lake  
7: natural grassland  
8: paddy field  
9: pond  
10: river  
11: rural residential  
12: shrub land  
13: traffic land  
14: urban residential
```

---

## **Installation**

```bash
pip install transformers torch pillow gradio
```

---

## **Example Inference Code**

```python
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
```

---

## **Applications**

* **Urban Development Planning**
* **Agricultural Monitoring**
* **Land Use and Land Cover (LULC) Mapping**
* **Disaster Management and Flood Risk Analysis**
