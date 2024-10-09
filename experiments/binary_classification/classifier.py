import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, simple = False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self.simple_fc = nn.Linear(input_dim, 1)
        self.simple = simple
        

    def forward(self, x):
        if not self.simple:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            # x = self.sigmoid(self.fc4(x))
        else:
            x = self.simple_fc(x)
            # x = self.sigmoid(self.simple_fc(x))
        return x