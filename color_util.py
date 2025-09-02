import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ColorTransform(nn.Module):
    def __init__(self, para_path):
        super(ColorTransform, self).__init__()
        file = np.load(para_path, allow_pickle=True)
        self.degree = file['d']
        weight = torch.from_numpy(file['weight'])
        bias = torch.from_numpy(file['bias'])
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def poly_feature(self, x, degree=None):
        if degree is None:
            degree = self.degree
        n = x.shape[1]
        feature = [x.clone()]
        index = list(range(n))
        for d in range(1, degree):
            new = []
            k = 0
            for i in range(n):
                new.append(x[:, i:i + 1] * feature[-1][:, index[i]:])
                index[i] = k
                k = k + new[-1].shape[1]
            new = torch.cat(new, 1)
            feature.append(new)
        feature = torch.cat(feature, 1)
        return feature

    def forward(self, x):
        f = self.poly_feature(x)
        f = f.transpose(1, -1)
        #     pred = (f.unsqueeze(1) * weight.unsqueeze(0)).sum(2) + bias
        pred = torch.matmul(f, self.weight) + self.bias
        pred = pred.transpose(1, -1)
        return pred