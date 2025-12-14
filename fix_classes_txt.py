import torch

MODEL_PATH = "models/best_foodvision_model.pth"
CLASSES_PATH = "models/classes.txt"

# load model to get class count
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
num_classes = state_dict["fc.weight"].shape[0]

# load existing classes
classes = []
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = [l.strip() for l in f if l.strip()]

# pad missing classes
if len(classes) < num_classes:
    for i in range(len(classes), num_classes):
        classes.append(f"class_{i}")

# rewrite classes.txt
with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    for c in classes:
        f.write(c + "\n")

print(f"OK: classes.txt fixed → {len(classes)} classes")
