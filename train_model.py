import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


def main():
    # ================= CONFIG =================
    DATA_DIR = "data/unified_dataset"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")

    BATCH_SIZE = 96
    EPOCHS = 5
    LR = 1e-3
    IMG_SIZE = 224

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("🔥 Device:", DEVICE)

    # ================= TRANSFORMS =================
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ================= DATASETS =================
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    print("Train images:", len(train_dataset))
    print("Test images :", len(test_dataset))
    print("Classes     :", len(train_dataset.classes))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    # ================= MODEL =================
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model.to(DEVICE)

    # ================= TRAIN SETUP =================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")

    best_test_acc = 0.0
    os.makedirs("models", exist_ok=True)

    # ================= TRAINING =================
    for epoch in range(EPOCHS):
        # -------- TRAIN --------
        model.train()
        train_loss = 0.0
        train_correct = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for images, labels in loop:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_dataset)

        # -------- EVAL --------
        model.eval()
        test_correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()

        test_acc = test_correct / len(test_dataset)

        print("\n==============================")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")
        print(f"Test  Acc : {test_acc:.4f}")
        print("==============================")

        # -------- SAVE BEST MODEL --------
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "models/best_foodvision_model.pth")
            print("💾 Best model saved!")

    print("\n🔥 Training finished")
    print("🏆 Best Test Accuracy:", best_test_acc)


if __name__ == "__main__":
    main()
