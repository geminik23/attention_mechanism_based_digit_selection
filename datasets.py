
from torch.utils.data import Dataset
import torch

import numpy as np
class DigitSelection(Dataset):
    """
    This dataset creates the variable-length of digits and label. The label value is lowest number if index is even else highest number.
    data will be padded with 0 if the number of digits does not reach the max_samples
    """

    def __init__(self, dataset, max_samples=5):
        self.d= dataset
        self.max_samples = max_samples

    def __len__(self):
        return len(self.d)
    
    def __getitem__(self, i):
        nsamples = np.random.randint(1, self.max_samples)
        # selected digit index
        s = np.random.randint(0, len(self.d), size=nsamples)

        # x : [max_samples, 1*28*28]
        x = torch.stack([self.d[i][0] for i in s] + [torch.zeros((1,28,28)) for i in range(self.max_samples-nsamples)])
        # y : 1~9
        y = max([self.d[i][1] for i in s]) if i%2==0 else min([self.d[i][1] for i in s])
        return (x, i%2), y
        