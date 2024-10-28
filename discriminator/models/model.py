import torch
import torch.nn as nn

from discriminator.models.cnn_block import CNNBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=None):
        super(Discriminator, self).__init__()
        if features is None:
            self.features = [64, 128, 256, 512]
        else:
            self.features = features
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels * 2,
                out_channels=self.features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []

        for idx, feature in enumerate(self.features[1:]):
            layers.append(
                CNNBlock(
                    in_channels=self.features[idx],
                    out_channels=feature,
                    stride=1 if feature == self.features[-1] else 2,
                )
            )

        layers.append(
            nn.Conv2d(
                in_channels=self.features[-1],
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.concat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
