from torch import nn, tensor
from parameters import Parameters


class Generator(nn.Module):
    def __init__(self, param: Parameters):
        super().__init__()
        return

    def set_arch(self, param: Parameters) -> None:
        return

    def forward(self, x: tensor) -> tensor:
        return