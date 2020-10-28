from torch import nn, tensor
from parameters import Parameters

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x: tensor) -> tensor:
        return x