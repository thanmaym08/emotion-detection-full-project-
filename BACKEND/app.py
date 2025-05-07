from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image
import io
import json
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model, load_pretrained_model  # Import the correct functions

# Load label map
with open("utils/label_map.json", "r") as f:
    idx_to_class = json.load(f)

# Define transforms
def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=len(idx_to_class))  # Use your get_model method
model = load_pretrained_model(model, "saved_models/efficientnet_b4_best.pth")  # Load weights
model.to(device)
model.eval()

# Init app
app = FastAPI()

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    transforms = get_transforms()
    augmented = transforms(image=image_np)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        prediction = idx_to_class[str(predicted_idx)]

    return JSONResponse({"prediction": prediction})
