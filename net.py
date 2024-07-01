import torch
import numpy as np
import pandas as pd

# create NN model to the dataset of digit recognition
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input shape: (batch_size, 1, 28, 28)
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # Output shape: (batch_size, 32, 28, 28)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output shape: (batch_size, 32, 14, 14)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # Output shape: (batch_size, 64, 14, 14)
        # After another pooling, we get: (batch_size, 64, 7, 7)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 1024)  # Fully connected layer
        self.fc2 = torch.nn.Linear(1024, 10)  # Output layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x