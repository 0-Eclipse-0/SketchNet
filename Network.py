import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 4, 2, stride=2) # 128 x 128 x 4
        self.conv2 = nn.Conv2d(4, 16, 2, stride=2) # 32 x 32 x 16
        self.conv3 = nn.Conv2d(16, 32, 2, stride=2) # 8 x 8 x 32
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 251)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1) # Make 1 x 512 (prep linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
