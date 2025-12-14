import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Food classes (example Turkish cuisine list)
FOOD_CLASSES = [
    "dolma",
    "kebab",
    "baklava",
    "menemen",
    "pilav",
    "lahmacun",
    "manti",
    "sarma"
]

# Load pretrained EfficientNet model
def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Modify the classifier output to match the number of classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(FOOD_CLASSES))

    # Load model weights if trained model exists (for now we skip)
    model.eval()
    return model


# Image preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Predict function
def predict_image(model, image: Image.Image):
    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
    
    return FOOD_CLASSES[predicted.item()]
