
import torch
import torch.nn as nn
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(),
        )


    def forward(self, input):
        # input : [B, N, C, W, H]
        # flatten to [B, N, D]
        input = input.view(input.size(0), input.size(1), -1)
        return self.net(input)


##
# Scores
class DotScore(nn.Module):
    def __init__(self, H):
        super(DotScore, self).__init__()
        self.H = H
    
    def forward(self, s, context):
        """
        s : [B, N, D](B, T, H) shape
        context: [B, D] 
        score : [B, N]
        """
        # bmm([B, N, D], [B, D, 1]) -> [B, N]
        score = torch.bmm(s, context.unsqueeze(-1)) / np.sqrt(self.H)
        return score
