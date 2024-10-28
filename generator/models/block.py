import torch
import torch.nn as nn

from enums.activation import Activation


class Block(nn.Module):
    def __init__(
            self, in_channels, out_channels, down=True, activation=Activation.RELU, use_dropout=False
    ):
        super(Block, self).__init__()
        self.activation = activation.get_activation()
        self.block = nn.Sequential(
            (
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                )
                if down
                else nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(out_channels),
            self.activation,
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.use_dropout else x
