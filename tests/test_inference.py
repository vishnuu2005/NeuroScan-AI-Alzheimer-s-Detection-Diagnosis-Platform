import os
import io
import pytest
import torch
from PIL import Image
from models.resnet_model import get_resnet2d
from chatbot.inference import load_model, transform_image, predict

CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

def create_synthetic_image_file(tmp_path, width=224, height=224):
    file_path = str(tmp_path / "test_mri.png")
    img = Image.new("L", (width, height), color=120)
    img.save(file_path, format="PNG")
    return file_path

def test_adaptive_model_construction():
    # 1-channel model
    m1 = get_resnet2d(num_classes=4, in_channels=1, sequential_fc=False)
    assert m1.conv1.weight.shape == torch.Size([64, 1, 7, 7])
    assert isinstance(m1.fc, torch.nn.Linear)

    # 3-channel model with Sequential head
    m3 = get_resnet2d(num_classes=4, in_channels=3, sequential_fc=True)
    assert m3.conv1.weight.shape == torch.Size([64, 3, 7, 7])
    assert isinstance(m3.fc, torch.nn.Sequential)

def test_checkpoint_loading_strict():
    best_path = "models/resnet2d_alzheimers_best.pth"
    orig_path = "models/resnet2d_alzheimers.pth"

    if os.path.exists(best_path):
        m_best = load_model(best_path, num_classes=4)
        assert m_best.in_channels == 3
        assert m_best.training is False

    if os.path.exists(orig_path):
        m_orig = load_model(orig_path, num_classes=4)
        assert m_orig.in_channels == 1
        assert m_orig.training is False

def test_image_transformation_shapes(tmp_path):
    img_path = create_synthetic_image_file(tmp_path)
    
    # 1-channel transform
    t1 = transform_image(img_path, in_channels=1, image_size=224)
    assert t1.shape == torch.Size([1, 1, 224, 224])
    
    # 3-channel transform
    t3 = transform_image(img_path, in_channels=3, image_size=224)
    assert t3.shape == torch.Size([1, 3, 224, 224])

def test_inference_prediction_categories(tmp_path):
    model_path = "models/resnet2d_alzheimers_best.pth"
    if not os.path.exists(model_path):
        model_path = "models/resnet2d_alzheimers.pth"
    if not os.path.exists(model_path):
        pytest.skip("No model checkpoint present on disk.")

    model = load_model(model_path, num_classes=4)
    img_path = create_synthetic_image_file(tmp_path)
    tensor = transform_image(img_path, in_channels=model.in_channels)
    
    pred_class, confidence, all_probs, warning, elapsed_ms = predict(model, tensor, CLASSES, threshold=0.1)

    assert pred_class in CLASSES or pred_class == "Uncertain"
    assert 0.0 <= confidence <= 100.0
    assert len(all_probs) == 4
    for c in CLASSES:
        assert c in all_probs
        assert 0.0 <= all_probs[c] <= 100.0
    assert elapsed_ms > 0.0

def test_prediction_low_confidence_warning(tmp_path):
    model = get_resnet2d(num_classes=4, in_channels=3, sequential_fc=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Enforcing 1.0 (100%) threshold guarantees warning triggers
    pred_class, conf, all_probs, warning, elapsed_ms = predict(model, dummy_input, CLASSES, threshold=1.0)
    assert pred_class == "Uncertain"
    assert warning is not None
    assert "Low confidence" in warning
