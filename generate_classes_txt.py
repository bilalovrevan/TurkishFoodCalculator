from torchvision.datasets import ImageFolder

TRAIN_DIR = "unified_dataset/train"
OUTPUT_PATH = "models/classes.txt"

dataset = ImageFolder(root=TRAIN_DIR)
classes = dataset.classes

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for cls in classes:
        f.write(cls + "\n")

print(f"OK: classes.txt written with {len(classes)} classes")
