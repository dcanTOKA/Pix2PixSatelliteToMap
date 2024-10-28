import numpy as np
import pytest
from unittest.mock import patch
import torch
from utils.training_util import save_checkpoint, load_checkpoint


@pytest.fixture
def setup_model_optimizer():
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


@patch('torch.save')
def test_save_checkpoint(mock_save, setup_model_optimizer, epoch=1):
    model, optimizer = setup_model_optimizer
    save_checkpoint(model, optimizer, f'checkpoint_{epoch}.pth')
    args, kwargs = mock_save.call_args
    for key in args[0]['state_dict']:
        np.testing.assert_array_almost_equal(
            args[0]['state_dict'][key].numpy(), model.state_dict()[key].numpy()
        )
