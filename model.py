import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import FeatureExtractor


class AttentionSelectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, score):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.feature_extraction = FeatureExtractor(input_size, hidden_size, hidden_size)
        self.embedding = nn.Embedding(2, hidden_size)
        self.condition_net = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
        )

        self.score = score

        self.prediction = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_classes) 
        )

        self.weights = None

        pass
    
    def get_mask(self, x, false_value=0):
        """
        return the bool tensor [B, N]
        """

        dim = list(range(2,len(x.shape)))
        with torch.no_grad():
            mask = torch.sum((x != false_value), dim=dim) > 0
        return mask

    def forward(self, input):
        input, condition = input
        device = input.device

        # get mask : [B, N]
        mask = self.get_mask(input).to(device)

        # [B, N, C, W, H] -> [B, N, D]
        h = self.feature_extraction(input)


        # [B] -> [B, D] -> [B, D]
        # condition
        context = self.condition_net(self.embedding(condition))

        # [B, N, D] & [B, D] ->[B, N]
        scores = self.score(h, context)
        
        scores[~mask] = -1000.0
        # apply softmax -> [B, N, 1]
        self.weights = F.softmax(scores, dim=1) 
    
        # [B, N, D] * [B, N, 1] -> sum([B, N, D], dim=1) -> [B, D]
        final = (h*self.weights).sum(dim=1)
        return self.prediction(final)



 
