from enum import auto, Enum
import torch.nn as nn


class Activation(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    TANH = auto()

    def get_activation(self):
        if self == Activation.RELU:
            return nn.ReLU()
        elif self == Activation.LEAKY_RELU:
            return nn.LeakyReLU(0.2)
        elif self == Activation.TANH:
            return nn.Tanh()
