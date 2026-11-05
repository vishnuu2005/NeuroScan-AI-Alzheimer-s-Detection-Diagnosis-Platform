import os
import math
import torch
from models.resnet_model import get_resnet2d
from PIL import Image
import torchvision.transforms as transforms


# =======================
# Load Model
# =======================
def load_model(model_path, num_classes=4, dropout_prob=0.6, freeze_layers=True, device=None):
    """
    Load 2D ResNet18 pretrained model with 3-channel input.
    Safely loads checkpoint and moves model to GPU if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_resnet2d(num_classes=num_classes, dropout_prob=dropout_prob, freeze_layers=freeze_layers)

    # Load checkpoint safely
    checkpoint = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()
    return model


# =======================
# Image Preprocessing
# =======================
def transform_image(image_path, image_size=None):
    # Use the training image size (default 224) unless overridden
    if image_size is None:
        image_size = int(os.getenv('IMAGE_SIZE', '224'))

    img = Image.open(image_path).convert('L')  # grayscale only, no RGB conversion

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # single-channel normalization
    ])

    return transform(img).unsqueeze(0)



# =======================
# Uncertainty & Metrics
# =======================
def calculate_prediction_entropy(probs_tensor):
    """
    Calculate normalized Shannon entropy for model prediction probabilities.
    Returns float in [0.0, 1.0] where 0.0 represents complete certainty
    and 1.0 represents uniform uncertainty across all classes.
    """
    eps = 1e-12
    num_classes = probs_tensor.size(-1)
    if num_classes <= 1:
        return 0.0
    entropy = -torch.sum(probs_tensor * torch.log2(probs_tensor + eps), dim=-1).item()
    max_entropy = math.log2(num_classes)
    return float(min(max(entropy / max_entropy, 0.0), 1.0))


def get_top_predictions(all_probs, top_k=None):
    """
    Returns list of (class_name, probability_percentage) sorted by descending likelihood.
    """
    sorted_items = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        return sorted_items[:top_k]
    return sorted_items


# =======================
# Prediction
# =======================
def predict(model, image_tensor, classes, threshold=None):
    """
    Predict Alzheimer stage from MRI tensor.
    Returns predicted class, confidence %, all class probabilities, and warning if confidence low.
    """
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    # Default threshold: 0.3 (30%) — reduces 'Uncertain' responses for moderate confidences
    if threshold is None:
        try:
            threshold = float(os.getenv('PREDICTION_THRESHOLD', '0.3'))
        except Exception:
            threshold = 0.3

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence_tensor, pred_idx = torch.max(probs, dim=1)
        confidence = confidence_tensor.item() * 100
        predicted_class = classes[pred_idx.item()]
        all_probs = {classes[i]: float(probs[0, i]) * 100 for i in range(len(classes))}

    warning = None
    if confidence < threshold * 100:
        warning = "Low confidence prediction. Please consult a neurologist for further evaluation."
        predicted_class = "Uncertain"

    return predicted_class, confidence, all_probs, warning


def predict_detailed(model, image_tensor, classes, threshold=None):
    """
    Enhanced prediction returning rich diagnostic metrics including entropy and ranking.
    """
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    if threshold is None:
        try:
            threshold = float(os.getenv('PREDICTION_THRESHOLD', '0.3'))
        except Exception:
            threshold = 0.3

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence_tensor, pred_idx = torch.max(probs, dim=1)
        confidence = confidence_tensor.item() * 100
        predicted_class = classes[pred_idx.item()]
        all_probs = {classes[i]: float(probs[0, i]) * 100 for i in range(len(classes))}
        entropy = calculate_prediction_entropy(probs)
        top_ranked = get_top_predictions(all_probs)

    warning = None
    if confidence < threshold * 100:
        warning = "Low confidence prediction. Please consult a neurologist for further evaluation."
        predicted_class = "Uncertain"
    elif entropy > 0.85:
        warning = "High entropy across diagnostic stages. Independent clinical review advised."

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probs": all_probs,
        "entropy": entropy,
        "top_ranked": top_ranked,
        "warning": warning
    }