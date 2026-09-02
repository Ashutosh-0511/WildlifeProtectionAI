from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetLSTM(nn.Module):
    """Baseline temporal behaviour classifier.

    Input: [batch, sequence, channels, height, width]
    Output: [batch, classes]
    """

    def __init__(self, num_classes: int, hidden_size: int = 256, layers: int = 1, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.lstm = nn.LSTM(self.feature_dim, hidden_size, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        features = self.backbone(x.reshape(b * t, c, h, w)).reshape(b, t, self.feature_dim)
        sequence, _ = self.lstm(features)
        return self.classifier(self.dropout(sequence[:, -1]))
