import torch
import torch.nn as nn

class MLPModel(nn.Module):

    def __init__(self, input_size):

        super(MLPModel, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(input_size, 128),

            nn.ReLU(),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 1)

        )

    def forward(self, x):

        # Flatten input
        x = x.view(x.size(0), -1)

        return self.model(x)