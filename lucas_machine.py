import torch.nn as nn
import torch.nn.functional as F
import torch


class LucasMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            # nn.ReLU(),
            # nn.Linear(4, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

