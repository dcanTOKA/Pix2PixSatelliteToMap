import pytest
import torch
from torch import Tensor
from discriminator.models.cnn_block import CNNBlock


@pytest.fixture
def random_input() -> Tensor:
    return torch.rand(1, 3, 64, 64)


@pytest.fixture()
def cnn_block() -> CNNBlock:
    return CNNBlock(in_channels=3, out_channels=16, stride=2)


def test_cnn_block_output_shape(random_input: Tensor, cnn_block: CNNBlock):
    output = cnn_block(random_input)

    expected_output_shape = (1, 16, 32, 32)

    assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, but got {output.shape}"
