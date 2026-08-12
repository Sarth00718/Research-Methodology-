import torch.nn as nn
from torchvision import models

def get_alexnet(num_classes):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    # AlexNet classifier has 6 layers, the last one is Linear(4096, 1000)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model
