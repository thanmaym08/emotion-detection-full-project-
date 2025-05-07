import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError  # ✅ Added

class CustomFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(root_dir)))}

        for class_name in self.class_to_idx:
            class_dir = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_file))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"[Skipped Corrupt] {self.images[idx]}")
            return self.__getitem__((idx + 1) % len(self.images))  # Skip to next image

        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        label = self.labels[idx]
        return image, label
