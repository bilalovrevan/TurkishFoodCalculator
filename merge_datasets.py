import os
import shutil
import random

food101_root = "dataset"
turkish_root = "data/turkish_hf"
kaggle_turkish_root = "data/kaggle_turkish_food"

output_root = "unified_dataset"
train_dir = os.path.join(output_root, "train")
test_dir = os.path.join(output_root, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def simple_split(data_list, test_ratio=0.2):
    random.shuffle(data_list)
    split_idx = int(len(data_list) * (1 - test_ratio))
    return data_list[:split_idx], data_list[split_idx:]

def merge_dataset(src_root, prefix=""):
    print(f"Merging: {src_root}")
    
    for class_name in os.listdir(src_root):
        class_path = os.path.join(src_root, class_name)
        if not os.path.isdir(class_path):
            continue

        clean_name = prefix + class_name.replace(" ", "_").replace("-", "_")
        os.makedirs(os.path.join(train_dir, clean_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, clean_name), exist_ok=True)

        images = [
            os.path.join(class_path, img)
            for img in os.listdir(class_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) < 1:
            continue

        train_imgs, test_imgs = simple_split(images)

        for img in train_imgs:
            shutil.copy(img, os.path.join(train_dir, clean_name))
        for img in test_imgs:
            shutil.copy(img, os.path.join(test_dir, clean_name))

        print(f"Class {clean_name}: {len(train_imgs)} train | {len(test_imgs)} test")


merge_dataset(os.path.join(food101_root, "train"), prefix="food101_")
merge_dataset(turkish_root, prefix="turkish_")

if os.path.exists(kaggle_turkish_root):
    merge_dataset(kaggle_turkish_root, prefix="kaggle_")

if __name__ == "__main__":
    import os

    base = os.path.dirname(__file__)
    data_root = os.path.join(base, "data")

    food101_root = os.path.join(data_root, "food101", "food-101", "images")  # səndə fərqli ola bilər
    turkish_root = os.path.join(data_root, "turkish_food")
    turkish_hf_root = os.path.join(data_root, "turkish_hf")

    output_root = os.path.join(data_root, "unified_dataset")
    train_dir = os.path.join(output_root, "train")
    test_dir  = os.path.join(output_root, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Food101: artıq train/test varsa ona görə tənzimlə. Yoxdursa split edəcəyik.
    merge_dataset(os.path.join(food101_root), prefix="food101_")  # əgər food101-də birbaşa class folder-lardırsa

    merge_dataset(turkish_root, prefix="turkish_")
    merge_dataset(turkish_hf_root, prefix="turkish_hf_")

    print("Unified dataset is ready!")
