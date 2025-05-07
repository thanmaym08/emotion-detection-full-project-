 Live Face Emotion Detection using EfficientNet-B4

A deep learning-based real-time face emotion recognition system using the EfficientNet-B4 architecture. This project leverages a custom face expression dataset, strong data augmentation with Albumentations, and a fine-tuned EfficientNet-B4 backbone for high-accuracy emotion classification.

    GitHub Repo: Live_face_emotion_detection-EfficientNet_b4



    Features

    ✅ Custom dataset support (folder-structured)

    ✅ Corrupt image filtering & preprocessing

    ✅ Advanced data augmentation (Albumentations)

    ✅ Pretrained EfficientNet-B4 as the backbone

    ✅ Batch-wise training logs & epoch accuracy

    ✅ Saves trained model to disk for future inference

    ✅ Real-time prediction-ready (webcam/live feed support possible)



     Dataset Structure

Organize your raw data like this:

data/
├── train/
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   └── ...
└── validation/
    ├── angry/
    ├── happy/
    ├── sad/
    └── ...

After preprocessing, cleaned datasets will be saved to:

cleaned_data/
├── train/
├── validation/



git clone git@github.com:thanmaym08/Live_face_emotion_detection-EfficientNet_b4.git
cd Live_face_emotion_detection-EfficientNet_b4

# Create and activate virtual environment
python -m venv face_emotion_env
# On Windows
face_emotion_env\Scripts\activate
# On macOS/Linux
source face_emotion_env/bin/activate

# Install dependencies
pip install -r requirements.txt



 Preprocess & Clean Dataset

# Clean corrupt or unreadable images
python clean_and_copy_dataset.py

This will clean both training and validation images from data/ and save them to cleaned_data/.



 Train the Model

python train.py

    Epoch logs will show batch-level progress

    Trained model will be saved as:

saved_models/efficientnet_b4_emotion.pth



 Sample Training Output

📊 Epoch 1/10, Loss: 1.0524, Train Acc: 0.8415, Val Acc: 0.8823
📊 Epoch 2/10, Loss: 0.7642, Train Acc: 0.9137, Val Acc: 0.9614
...



Requirements

Install all dependencies using:

pip install -r requirements.txt

requirements.txt includes:

    torch

    torchvision

    efficientnet_pytorch

    albumentations

    opencv-python

    scikit-learn

    pillow

To install them all in one go:

pip install -r requirements.txt



    for the gpu based requirements are 

    # PyTorch with CUDA 11.8
torch==2.0.1+cu118
torchvision==0.15.2+cu118
torchaudio==2.0.2+cu118
# Use pip install with --extra-index-url for CUDA support
# EfficientNet model
efficientnet_pytorch==0.7.1
# Image augmentations
albumentations==1.3.0
opencv-python
# ML & Utilities
scikit-learn
pillow
numpy
matplotlib
tqdm

To install with GPU CUDA support:

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_gpu.txt



    Files in the Repository
File Name	Description
train.py	Training loop for EfficientNet-B4 model
model.py	EfficientNet-B4 model configuration
preprocess.py	Image preprocessing utilities
transforms.py	Data augmentation and transform functions
dataset_loader.py	Custom PyTorch Dataset class for loading data
clean_and_copy_dataset.py	Removes corrupt images and copies clean ones
preprocess_faces.py	(Optional) Face extraction logic if needed
.gitignore	Ignores environment files, cache, etc.



Pushing to GitHub 

git init
git add .
git commit -m "Initial commit: Live face emotion detection using EfficientNet B4"
git branch -M main
git remote add origin git@github.com:thanmaym08/Live_face_emotion_detection-EfficientNet_b4.git
git pull origin main --allow-unrelated-histories
git push -u origin main