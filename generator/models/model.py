import torch
import torch.nn as nn

from enums.activation import Activation
from generator.models.block import Block


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()
        self.leaky_relu = Activation.LEAKY_RELU
        self.relu = Activation.RELU
        self.tanh = Activation.TANH
        self.initial_down = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            self.leaky_relu.get_activation(),
        )

        self.down1 = Block(
            in_channels=features,
            out_channels=features * 2,
            down=True,
            activation=self.leaky_relu,
            use_dropout=False,
        )
        self.down2 = Block(
            in_channels=features * 2,
            out_channels=features * 4,
            down=True,
            activation=self.leaky_relu,
            use_dropout=False,
        )
        self.down3 = Block(
            in_channels=features * 4,
            out_channels=features * 8,
            down=True,
            activation=self.leaky_relu,
            use_dropout=False,
        )
        self.down4 = Block(
            in_channels=features * 8,
            out_channels=features * 8,
            down=True,
            activation=self.leaky_relu,
            use_dropout=False,
        )
        self.down5 = Block(
            in_channels=features * 8,
            out_channels=features * 8,
            down=True,
            activation=self.leaky_relu,
            use_dropout=False,
        )
        self.down6 = Block(
            in_channels=features * 8,
            out_channels=features * 8,
            down=True,
            activation=self.leaky_relu,
            use_dropout=False,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=features * 8,
                out_channels=features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.relu.get_activation(),
        )

        self.up1 = Block(
            in_channels=features * 8, down=False, out_channels=features * 8, use_dropout=True
        )
        self.up2 = Block(
            in_channels=features * 8 * 2, down=False, out_channels=features * 8, use_dropout=True
        )
        self.up3 = Block(
            in_channels=features * 8 * 2, down=False, out_channels=features * 8, use_dropout=True
        )
        self.up4 = Block(
            in_channels=features * 8 * 2, down=False, out_channels=features * 8, use_dropout=False
        )
        self.up5 = Block(
            in_channels=features * 8 * 2, down=False, out_channels=features * 4, use_dropout=False
        )
        self.up6 = Block(
            in_channels=features * 4 * 2, down=False, out_channels=features * 2, use_dropout=False
        )
        self.up7 = Block(
            in_channels=features * 2 * 2, down=False, out_channels=features, use_dropout=False
        )

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            self.tanh.get_activation()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        final = self.final_up(torch.cat([up7, d1], dim=1))

        return final
