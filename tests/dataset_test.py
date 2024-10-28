import pytest

from services.dataset import SatelliteMapDataset


@pytest.fixture()
def dataset_train_directory() -> str:
    return "../data/original/train"


@pytest.fixture()
def dataset_val_directory() -> str:
    return "../data/original/val"


def test_dataset_initialization(dataset_train_directory, dataset_val_directory):
    train_dataset = SatelliteMapDataset(root_dir=dataset_train_directory)
    val_dataset = SatelliteMapDataset(root_dir=dataset_val_directory)

    assert len(train_dataset) > 0, "Train dataset is empty"
    assert len(val_dataset) > 0, "Train dataset is empty"


def test_getitem(dataset_train_directory):
    dataset = SatelliteMapDataset(root_dir=dataset_train_directory)

    input_image, target_image = dataset[0]

    assert input_image.shape == (3, 256, 256), f"Input image shape is incorrect: {input_image.shape}"
    assert target_image.shape == (3, 256, 256), f"Target image shape is incorrect: {target_image.shape}"
