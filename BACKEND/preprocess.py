import os
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

def preprocess_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)

        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert("RGB")
            face = mtcnn(img)

            if face is not None:
                face_img = face.permute(1, 2, 0).byte().cpu().numpy()
                cv2.imwrite(os.path.join(output_class_path, img_name), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

# Example usage
# preprocess_images("raw_dataset", "data/train")
