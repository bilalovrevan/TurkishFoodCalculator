import sys
import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from nutrition_data import NUTRITION_DATA

# ================= CONFIG =================
MODEL_PATH = "models/best_foodvision_model.pth"
CLASSES_PATH = "models/classes.txt"
THRESHOLD = 0.40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================


def humanize_class_name(class_name: str) -> str:
    """
    Converts:
    Yaprak_Sarma -> Yaprak Sarma
    apple_pie -> Apple Pie
    """
    return class_name.replace("_", " ").title()


# ---------- usage ----------
if len(sys.argv) < 2:
    print("Usage: python predict.py path/to/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("Image not found")
    sys.exit(1)

# ---------- load classes ----------
if not os.path.exists(CLASSES_PATH):
    print("classes.txt not found")
    sys.exit(1)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = [l.strip() for l in f if l.strip()]

# ---------- load model ----------
if not os.path.exists(MODEL_PATH):
    print("Model file not found")
    sys.exit(1)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

num_classes = state_dict["fc.weight"].shape[0]

if len(classes) != num_classes:
    print(f"CLASS MISMATCH: model={num_classes}, classes.txt={len(classes)}")
    sys.exit(1)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# ---------- transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- image ----------
try:
    img = Image.open(image_path).convert("RGB")
except Exception as e:
    print("Image could not be opened:", e)
    sys.exit(1)

img = transform(img).unsqueeze(0).to(DEVICE)

# ---------- prediction ----------
with torch.no_grad():
    outputs = model(img)
    probs = F.softmax(outputs, dim=1)

    max_prob, max_idx = torch.max(probs, dim=1)
    confidence = max_prob.item()

# ---------- result ----------
print("\n🔔 Notification:")

if confidence < THRESHOLD:
    print("❌ This is not a food image.")
    print("👉 Please upload a food photo.")
    print(f"(Confidence: {confidence*100:.2f}%)")

else:
    raw_food_name = classes[max_idx.item()]
    food_name = humanize_class_name(raw_food_name)

    print("✅ Food detected successfully!\n")
    print(f"🍽 Food Name: {food_name}")
    print(f"🎯 Confidence: {confidence*100:.2f}%")

    if raw_food_name in NUTRITION_DATA:
        nut = NUTRITION_DATA[raw_food_name]
        print("\n🥗 Nutritional Values (per serving):")
        print(f"🔥 Calories: {nut['calories']} kcal")
        print(f"💪 Protein: {nut['protein']} g")
        print(f"🧈 Fat: {nut['fat']} g")
        print(f"🍞 Carbs: {nut['carbs']} g")
    else:
        print("\n⚠️ Nutrition data not available for this food.")
