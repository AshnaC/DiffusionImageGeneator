import math

import torch
import torch.nn as nn
import random
import numpy as np


# Takes in class indices and return label encoding
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, embedding_dim, prob):
        super(LabelEmbedder, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        # self.default_encoding = nn.Parameter(torch.randn(1, embedding_dim))
        self.default_encoding = nn.Parameter(torch.zeros(1, embedding_dim))

        self.prob = prob

    def forward(self, class_indices, is_default=False):
        # If we are planning for class unconditional default = False
        # Then we choose random indices to attach correct class embeddings to
        # For the rest of the samples we put default embeddings
        indices = torch.Tensor(self.choose_numbers(class_indices) if not is_default else range(0, len(class_indices))).int()

        embedded_labels = self.embedding(class_indices)
        embedded_labels[indices] = self.default_encoding
        return embedded_labels

    def choose(self):
        choices = [True, False]
        weights = [self.prob, 1 - self.prob]
        result = random.choices(choices, weights=weights, k=1)
        return result[0]

    def choose_numbers(self, class_indices):
        index = range(0, len(class_indices))
        n = math.floor(len(index) * self.prob)
        chosen_numbers = random.sample(index, n)
        return chosen_numbers



