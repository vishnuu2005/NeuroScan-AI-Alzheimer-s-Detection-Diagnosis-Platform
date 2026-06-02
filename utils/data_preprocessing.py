import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

class GrayToRGB:
    """Custom transform to convert grayscale PIL image to RGB by duplicating channels."""
    def __call__(self, img):
        return img.convert("RGB")

def get_data_loaders(batch_size=32, image_size=224, use_augmented=True):
    """
    Returns training and testing data loaders for Alzheimer MRI images.
    Grayscale images are converted to RGB for pretrained ResNet.
    """

    # Dataset Paths on D: drive
    original_train_dir = r"D:/AlzeigmersChatBot/AlzeigmersChatBot/data/train"
    augmented_train_dir = r"D:/AlzeigmersChatBot/AlzeigmersChatBot/data/train_aug"
    test_dir = r"D:/AlzeigmersChatBot/AlzeigmersChatBot/data/test"

    # Choose training directory
    if use_augmented and os.path.exists(augmented_train_dir):
        print("Using augmented + balanced training data.")
        train_dir = augmented_train_dir
    else:
        print("Using original training data.")
        train_dir = original_train_dir

    # Transformations
    train_transform = transforms.Compose([
        GrayToRGB(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    test_transform = transforms.Compose([
        GrayToRGB(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Create Datasets
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_dataset.classes
