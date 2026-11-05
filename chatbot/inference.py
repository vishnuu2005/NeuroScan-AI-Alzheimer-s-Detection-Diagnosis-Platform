import os
import time
import torch
from PIL import Image
import torchvision.transforms as transforms

try:
    from models.resnet_model import get_resnet2d
except ImportError:
    from ..models.resnet_model import get_resnet2d


# =======================
# Adaptive Model Loader
# =======================
def load_model(model_path, num_classes=4, dropout_prob=0.6, freeze_layers=True, device=None):
    """
    Safely loads 2D ResNet-18 checkpoint with adaptive channel detection and strict validation.
    Auto-detects whether the checkpoint was trained on grayscale (1-ch) or RGB (3-ch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    # Inspect checkpoint conv1 input channels
    if 'conv1.weight' in checkpoint:
        in_channels = checkpoint['conv1.weight'].shape[1]
    else:
        in_channels = 3

    # Inspect classifier head architecture (sequential with dropout vs direct linear)
    sequential_fc = 'fc.1.weight' in checkpoint

    model = get_resnet2d(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_prob=dropout_prob,
        freeze_layers=freeze_layers,
        sequential_fc=sequential_fc
    )

    model.load_state_dict(checkpoint, strict=True)
    model.in_channels = in_channels
    model.sequential_fc = sequential_fc
    model.to(device)
    model.eval()
    try:
        warmup_model(model, in_channels=in_channels)
    except Exception:
        pass
    return model


def warmup_model(model, in_channels=None, image_size=None):
    """
    Executes a synthetic forward pass to warm up PyTorch CPU/CUDA kernels and eliminate first-request latency.
    """
    if in_channels is None:
        in_channels = getattr(model, 'in_channels', 3)
    if image_size is None:
        try:
            image_size = int(os.getenv('IMAGE_SIZE', '224'))
        except Exception:
            image_size = 224

    device = next(model.parameters()).device
    dummy_input = torch.zeros((1, in_channels, image_size, image_size), device=device)
    model.eval()
    with torch.inference_mode():
        _ = model(dummy_input)
    return True


# =======================
# Image Preprocessing
# =======================
def transform_image(image_path, in_channels=None, image_size=None):
    """
    Transforms MRI image file into normalized PyTorch tensor.
    Adapts automatically to 1-channel or 3-channel input representations.
    Ensures safe file closure via Pillow context manager.
    """
    if image_size is None:
        image_size = int(os.getenv('IMAGE_SIZE', '224'))

    if in_channels is None:
        in_channels = int(os.getenv('IN_CHANNELS', '3'))

    with Image.open(image_path) as raw_img:
        if in_channels == 1:
            img = raw_img.convert('L')
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            img = raw_img.convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        tensor = transform(img).unsqueeze(0)
    return tensor


# =======================
# Prediction Pipeline
# =======================
def predict(model, image_tensor, classes, threshold=None):
    """
    Predicts Alzheimer stage from MRI tensor using accelerated inference mode.
    Returns predicted class, confidence %, all class probabilities, warning, and elapsed latency in ms.
    """
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    if threshold is None:
        try:
            threshold = float(os.getenv('PREDICTION_THRESHOLD', '0.30'))
        except Exception:
            threshold = 0.30

    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence_tensor, pred_idx = torch.max(probs, dim=1)
        confidence = confidence_tensor.item() * 100
        predicted_class = classes[pred_idx.item()]
        all_probs = {classes[i]: round(float(probs[0, i]) * 100, 2) for i in range(len(classes))}
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    warning = None
    if confidence < threshold * 100:
        warning = "Low confidence prediction. Please consult a qualified neurologist for professional clinical evaluation."
        predicted_class = "Uncertain"

    return predicted_class, round(confidence, 2), all_probs, warning, elapsed_ms
