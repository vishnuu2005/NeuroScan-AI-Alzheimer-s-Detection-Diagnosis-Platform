import torchvision.models as models
import torch.nn as nn


def get_resnet2d(num_classes=4, dropout_prob=0.6, freeze_layers=True):
    model = models.resnet18(pretrained=False)

    # Match the saved checkpoint — grayscale input (1 channel)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_prob),
        nn.Linear(in_features, num_classes)
    )

    if freeze_layers:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'conv1' not in name:
                param.requires_grad = False

    return model

