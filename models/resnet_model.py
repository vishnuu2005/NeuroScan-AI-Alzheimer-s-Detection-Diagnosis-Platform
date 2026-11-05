import torchvision.models as models
import torch.nn as nn


def get_resnet2d(num_classes=4, in_channels=3, dropout_prob=0.6, freeze_layers=True, sequential_fc=True):
    """
    Constructs a 2D ResNet-18 model with configurable input channels and classifier head.
    Supports both single-channel (grayscale) and three-channel (RGB) MRI representations.
    """
    model = models.resnet18(weights=None)

    # Configure input layer according to channel count
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Configure classifier head
    in_features = model.fc.in_features
    if sequential_fc:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    if freeze_layers:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'conv1' not in name:
                param.requires_grad = False

    return model
