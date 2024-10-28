import pytest
import torch
from torch import Tensor

from enums.activation import Activation
from generator.models.block import Block


@pytest.fixture()
def random_input_down() -> Tensor:
    return torch.randn((1, 3, 256, 256))


@pytest.fixture()
def random_input_up() -> Tensor:
    return torch.randn((1, 64, 2, 2))


@pytest.fixture()
def block_with_down() -> Block:
    return Block(in_channels=3, out_channels=64, down=True, activation=Activation.LEAKY_RELU)


@pytest.fixture()
def block_with_up() -> Block:
    return Block(in_channels=64, out_channels=32, down=False, activation=Activation.LEAKY_RELU)


def test_block_forward_down(random_input_down, block_with_down):
    output = block_with_down(random_input_down)

    expected_output_shape = (1, 64, 128, 128)

    assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, but got {output.shape}"


def test_block_forward_up(random_input_up, block_with_up):
    output = block_with_up(random_input_up)

    expected_output_shape = (1, 32, 4, 4)

    assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, but got {output.shape}"
