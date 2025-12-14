import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ================= CONFIG =================
DATA_DIR = "unified_dataset/train"
MODEL_OUT = "models/best_foodvision_model.pth"
CLASSES_OUT = "models/classes.txt"

BATCH_SIZE = 64
EPOCHS = 1
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================


def main():
    # ================= TRANSFORMS =================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # ==============================================

    # ================= DATASET ====================
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6  # 🔥 WINDOWS FIX
    )

    num_classes = len(train_dataset.classes)
    print("Classes:", num_classes)
    # ==============================================

    # ================= SAVE CLASSES =================
    os.makedirs("models", exist_ok=True)
    with open(CLASSES_OUT, "w", encoding="utf-8") as f:
        for cls in train_dataset.classes:
            f.write(cls + "\n")
    print("✅ classes.txt saved")
    # ===============================================

    # ================= MODEL =======================
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # ===============================================

    # ================= TRAIN =======================
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
    # ===============================================

    # ================= SAVE MODEL ==================
    torch.save(model.state_dict(), MODEL_OUT)
    print("✅ Model saved:", MODEL_OUT)
    # ===============================================


if __name__ == "__main__":
    main()
