import pytest
import torch
from torch import Tensor
from generator.models.model import Generator


@pytest.fixture()
def random_input() -> Tensor:
    return torch.randn((1, 3, 256, 256))


@pytest.fixture()
def generator() -> Generator:
    return Generator()


def test_generator_forward(random_input, generator):
    output = generator(random_input)

    expected_output_shape = (1, 3, 256, 256)

    assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, but got {output.shape}"
