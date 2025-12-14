from datasets import load_dataset
import os
from PIL import Image

dataset_name = "alpsahin/Turkish-Food-Dataset-v3"

print(f"Downloading dataset: {dataset_name}")

dataset = load_dataset(dataset_name, split="train")

output_dir = "data/turkish_hf"
os.makedirs(output_dir, exist_ok=True)

for idx, item in enumerate(dataset):
    img = item["raw-img"]              # This is already a PIL image
    cls = item["class"]                # Class label

    # Create class directory
    class_dir = os.path.join(output_dir, cls)
    os.makedirs(class_dir, exist_ok=True)

    # Save image
    save_path = os.path.join(class_dir, f"{idx}.jpg")
    img.save(save_path)

    if idx % 500 == 0:
        print(f"Saved {idx} images...")

print("Dataset fully saved!")
