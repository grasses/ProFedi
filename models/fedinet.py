import torch.nn as nn


class FeDINet(nn.Module):
    def __init__(self):
        super(FeDINet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(500, 256),
            nn.Linear(256, 32),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.model(x)
        return out