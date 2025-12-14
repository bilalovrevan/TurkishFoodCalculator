import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_foodvision_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.txt")
# ===========================================

TOP_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD CLASSES =================
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = [l.strip() for l in f if l.strip()]
# ===============================================

# ================= LOAD MODEL ===================
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
num_classes = state_dict["fc.weight"].shape[0]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
# ===============================================

# ================= TRANSFORM ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ===============================================

# ================= UI ===========================
st.set_page_config(page_title="FoodVisionAI", layout="centered")
st.title("🍽️ FoodVisionAI")
st.write("Upload a food image and get predictions")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(TOP_K)

    st.subheader("Prediction results:")
    for i in range(TOP_K):
        idx = top_idxs[0][i].item()
        label = classes[idx]
        confidence = top_probs[0][i].item() * 100
        st.write(f"**{i+1}. {label} — {confidence:.2f}%**")
# ===============================================
