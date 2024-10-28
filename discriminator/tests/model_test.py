from typing import Tuple, List

import pytest
import torch
from torch import Tensor
from discriminator.models.model import Discriminator


@pytest.fixture()
def random_input_and_output() -> Tuple[Tensor, Tensor]:
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)

    return x, y


@pytest.fixture()
def discriminator_features() -> List:
    return [64, 128, 256, 512]


@pytest.fixture()
def discriminator(discriminator_features) -> Discriminator:
    return Discriminator(in_channels=3, features=discriminator_features)


def test_discriminator_initialization(discriminator):
    assert isinstance(discriminator, Discriminator), "Discriminator instance cannot be created properly."


def test_discriminator_forward(discriminator, random_input_and_output):
    output = discriminator(random_input_and_output[0], random_input_and_output[1])

    expected_output_shape = (1, 1, 30, 30)

    assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, but got {output.shape}"
