from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

data_dir = "data/unified_dataset"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = ImageFolder(root=f"{data_dir}/train", transform=transform)
test_ds  = ImageFolder(root=f"{data_dir}/test", transform=transform)

print("Train size:", len(train_ds))
print("Test size:", len(test_ds))
print("Num classes:", len(train_ds.classes))
print("First 20 classes:", train_ds.classes[:20])

# mini batch check
dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
x, y = next(iter(dl))
print("Batch x:", x.shape)
print("Batch y:", y.shape)
