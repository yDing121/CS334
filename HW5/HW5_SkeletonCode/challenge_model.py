'''
Challenge - Model
    Constructs a pytorch model for a convolutional neural network
'''
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Challenge(nn.Module):
    def __init__(self):
        super(Challenge, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2),
        )

        # Global Average Pooling instead of flatten
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Additional Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')  # He initialization for Conv2d
            if layer.bias is not None:
                init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            init.xavier_normal_(layer.weight)  # Xavier initialization for Linear layers
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        z = self.conv_layers(x)
        z = self.global_avg_pool(z)
        z = self.fc_layers(z)
        return z
