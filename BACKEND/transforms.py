import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 380  # EfficientNet-B4 expected input size

def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),  # 🔥 Add this line
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),  # 🔥 Add this line too
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
