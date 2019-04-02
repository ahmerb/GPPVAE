import torch
from torch import nn
import torch.nn.functional as F


class UnobservedFeatureVectors(nn.Module):
    def __init__(self, feature_ids, num_vectors, num_features):
        super(UnobservedFeatureVectors, self).__init__()
        self.feature_vectors = nn.Parameter(torch.randn(num_vectors, num_features))

    def forward(self, test_feature_ids):
        test_feature_vectors = F.embedding(test_feature_ids, self.feature_vectors)
        return test_feature_vectors
