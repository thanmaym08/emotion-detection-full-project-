import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EmotionEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b4")
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):   
        return self.model(x)

def get_model(num_classes):
    return EmotionEfficientNet(num_classes)

# Optional: add this for loading pretrained weights if needed
def load_pretrained_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
