import os
import shutil

DATASET_ROOT = "data/food101/food-101"
IMAGES_FOLDER = os.path.join(DATASET_ROOT, "images")
META_FOLDER = os.path.join(DATASET_ROOT, "meta")

OUTPUT_DIR = "dataset"

# Create train/test root folders
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

# Load class list
with open(os.path.join(META_FOLDER, "classes.txt"), "r") as f:
    classes = [c.strip() for c in f.readlines()]

# Create class folders
for c in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, "train", c), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test", c), exist_ok=True)

def process_split(split):
    txt_file = os.path.join(META_FOLDER, f"{split}.txt")
    with open(txt_file, "r") as f:
        lines = [x.strip() for x in f.readlines()]

    for line in lines:
        class_name, image_name = line.split("/")
        src = os.path.join(IMAGES_FOLDER, class_name, image_name + ".jpg")
        dst = os.path.join(OUTPUT_DIR, split, class_name, image_name + ".jpg")

        if os.path.exists(src):
            shutil.copy(src, dst)

    print(f"{split} split completed.")

process_split("train")
process_split("test")

print("Food-101 dataset preparation completed.")
