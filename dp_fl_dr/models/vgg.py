import torch.nn as nn
from torchvision import models

def get_vgg16(num_classes):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # VGG16 classifier has 6 layers, the last one is Linear(4096, 1000)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model
