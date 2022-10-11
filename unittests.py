import unittest
import torch
from modules import FeatureExtractor
from model import AttentionSelectionModel

class DigitSelectionTestCase(unittest.TestCase):
    def test_feature_extraction(self):
        B = 3
        N = 3
        D = 28*28
        OUT_D = 128
        x = torch.randn((B, N, 1, 28, 28))
        net = FeatureExtractor(D, OUT_D, OUT_D)
        y = net(x)
        self.assertEqual(y.shape, torch.Size([B,N,OUT_D]))

    def test_mask(self):
        # B(batch), N(num_samples), C, W, H
        x = torch.zeros((2, 2, 1, 28, 28))
        x[0, 1, :] = 1
        mask = AttentionSelectionModel.get_mask(None, x)

        expected = torch.tensor([[False, True], [False, False]]).bool()
        self.assertTrue((mask == expected).sum() == 4)
        
        


        

if __name__ == '__main__':
    unittest.main()        