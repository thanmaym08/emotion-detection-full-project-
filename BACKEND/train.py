import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset_loader import CustomFaceDataset
from transforms import get_train_transforms, get_validation_transforms
from model import EmotionEfficientNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Use cleaned dataset paths
train_dataset = CustomFaceDataset(root_dir="cleaned_data/train", transform=get_train_transforms())
val_dataset = CustomFaceDataset(root_dir="cleaned_data/validation", transform=get_validation_transforms())

# ✅ Show dataset sizes
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Validation Dataset Size: {len(val_dataset)}")

# ✅ Set num_workers=0 to avoid freezing issues
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

model = EmotionEfficientNet(num_classes=len(train_dataset.class_to_idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 10
print("🚀 Starting training...")

for epoch in range(EPOCHS):
    print(f"\n🌀 Epoch {epoch+1}/{EPOCHS} begins...")
    model.train()
    running_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, (images, labels) in enumerate(train_loader):
        try:
            print(f"🔁 Training batch {batch_idx + 1}/{len(train_loader)}")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            continue

    train_acc = accuracy_score(all_labels, all_preds)

    # ✅ Validation step
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"📊 Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# ✅ Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/efficientnet_b4_emotion.pth")
print("💾 Model saved to saved_models/efficientnet_b4_emotion.pth")
