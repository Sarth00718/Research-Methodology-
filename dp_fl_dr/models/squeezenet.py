import torch.nn as nn
from torchvision import models

def get_squeezenet1_1(num_classes):
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    # SqueezeNet classifier is a sequential block.
    # classifier[1] is a Conv2d(512, 1000, kernel_size=(1,1))
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    # We must also update the num_classes attribute
    model.num_classes = num_classes
    return model
