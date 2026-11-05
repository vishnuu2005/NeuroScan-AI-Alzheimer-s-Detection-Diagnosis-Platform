"""
Unit tests for NeuroScan neural network architectures.
"""

import unittest
try:
    import torch
    from models.resnet_model import get_resnet2d
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestResNet2DArchitecture(unittest.TestCase):
    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in the current environment.")

    def test_model_initialization(self):
        """Verify model instantiates with 4 default classes and 1-channel input."""
        model = get_resnet2d(num_classes=4, dropout_prob=0.5, freeze_layers=False)
        self.assertIsNotNone(model)
        self.assertEqual(model.conv1.in_channels, 1)
        self.assertEqual(model.conv1.out_channels, 64)

    def test_forward_pass_shape(self):
        """Verify forward pass returns tensor of shape (batch_size, num_classes)."""
        model = get_resnet2d(num_classes=4, dropout_prob=0.5, freeze_layers=False)
        model.eval()
        dummy_input = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (2, 4))

    def test_custom_num_classes(self):
        """Verify model handles arbitrary class counts properly."""
        model = get_resnet2d(num_classes=6, dropout_prob=0.3, freeze_layers=False)
        dummy_input = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (1, 6))


if __name__ == '__main__':
    unittest.main()
