'''
CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.cnn import CNN
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: define each layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2
        )

        self.fc1 = nn.Linear(
            in_features=512,
            out_features=64
        )

        self.fc2 = nn.Linear(
            in_features=64,
            out_features=32
        )

        self.fc3 = nn.Linear(
            in_features=32,
            out_features=5
        )

        self.init_weights()
    
    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)

        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(fc.weight, 0.0, 1/fc.in_features)
            nn.init.constant_(conv.bias, 0.0)


        
    def forward(self, x):
        N, C, H, W = x.shape

        # TODO: forward pass

        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = z.view(N, -1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)

        
        return z
