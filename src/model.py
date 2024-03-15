import torch
import torch.nn as nn
import config
import torch
from torch import nn
import torchvision.models as models

import torch.nn as nn
import torchvision.models as models

class ObjectDetector(nn.Module):
    def __init__(self, num_classes=3):
        super(ObjectDetector, self).__init__()

        # Backbone (pre-trained ResNet50)
        self.backbone = models.resnet50(pretrained=True)
        # Modify the first layer to accept 3 channels instead of 3
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Additional convolutional layers
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        # Average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 416, 416).to(config.DEVICE)  # Batch size 1, 3 channels, 416x416 image

    model = ObjectDetector().to(config.DEVICE)
    output = model(dummy_input)

    print(output.size())
