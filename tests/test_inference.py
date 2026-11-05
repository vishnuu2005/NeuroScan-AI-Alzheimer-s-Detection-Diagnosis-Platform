"""
Unit tests for inference, prediction formatting, and uncertainty estimation.
"""

import unittest
try:
    import torch
    import torch.nn as nn
    from chatbot.inference import predict
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class MockModel(nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

        def forward(self, x):
            return self.logits.repeat(x.size(0), 1)


class TestInferencePipeline(unittest.TestCase):
    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not available in test environment.")
        self.classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

    def test_predict_high_confidence(self):
        """Verify clear winner predicts high confidence without warning."""
        mock_model = MockModel([[0.1, 0.2, 5.0, 0.3]])
        dummy_tensor = torch.zeros(1, 1, 224, 224)
        pred, conf, all_probs, warn = predict(mock_model, dummy_tensor, self.classes, threshold=0.3)
        self.assertEqual(pred, 'NonDemented')
        self.assertGreater(conf, 90.0)
        self.assertIsNone(warn)
        self.assertIn('NonDemented', all_probs)

    def test_predict_low_confidence_uncertainty(self):
        """Verify low-confidence prediction triggers Uncertain label and warning."""
        mock_model = MockModel([[0.25, 0.25, 0.26, 0.24]])
        dummy_tensor = torch.zeros(1, 1, 224, 224)
        pred, conf, all_probs, warn = predict(mock_model, dummy_tensor, self.classes, threshold=0.5)
        self.assertEqual(pred, 'Uncertain')
        self.assertIsNotNone(warn)
        self.assertIn('Low confidence', warn)


if __name__ == '__main__':
    unittest.main()
