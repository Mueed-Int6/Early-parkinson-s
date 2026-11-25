# Early-parkinson's
import os
import random
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from google.colab import files

# 1. Upload archive (1).zip from your laptop (if not already in /content)
zip_path = "/content/archive (1).zip"
extract_dir = "/content/parkinson_data"

if not os.path.exists(zip_path):
    print("Please select 'archive (1).zip' from your laptop:")
    uploaded = files.upload()
    for fname in uploaded.keys():
        if fname != "archive (1).zip":
            os.rename(fname, zip_path)

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    os.system(f'unzip -q "{zip_path}" -d "{extract_dir}"')

# 2. Locate spiral/training and spiral/testing automatically
candidates = [
    os.path.join(extract_dir, "drawings", "spiral"),
    os.path.join(extract_dir, "spiral"),
]
spiral_dir = next((p for p in candidates if os.path.exists(os.path.join(p, "training"))), None)
if spiral_dir is None:
    for root, dirs, _ in os.walk(extract_dir):
        if os.path.basename(root) == "spiral" and "training" in dirs:
            spiral_dir = root
            break

assert spiral_dir is not None, f"Could not find spiral/training inside {extract_dir}"

train_path = os.path.join(spiral_dir, "training")
test_path = os.path.join(spiral_dir, "testing")
print("Using spiral directory:", spiral_dir)

# 3. Reproducibility & Device Setup
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 4. Transforms (no filename filtering so V01HE01.png, V01PE01.png, etc. all load)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 5. Custom Dataset Class
class ParkinsonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        class_map = {
            "healthy": 0,
            "parkinson": 1
        }

        for class_name, label_idx in class_map.items():
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                continue
            for file in sorted(os.listdir(class_path)):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(class_path, file))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# 6. Datasets & DataLoaders
train_dataset = ParkinsonDataset(train_path, transform=train_transform)
test_dataset = ParkinsonDataset(test_path, transform=eval_transform)

print("Train images loaded:", len(train_dataset))
print("Test images loaded:", len(test_dataset))
assert len(train_dataset) > 0 and len(test_dataset) > 0, "Dataset is empty; check folder structure."

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 7. Model Initialization & Binary Classification Head
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)
model = model.to(device)
print("Model ready on:", device)

# 8. Loss Function, Optimizer & Training Loop
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, (correct / total) * 100.0

num_epochs = 15
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    test_loss, test_acc = run_epoch(test_loader, training=False)
    print(
        f"Epoch [{epoch}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
    )

# 9. Save Model Weights
save_path = "/content/parkinson_efficientnet_b0.pth"
torch.save(model.state_dict(), save_path)
print("Saved model weights to:", save_path)
