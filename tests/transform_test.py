import numpy as np
import pytest

from PIL import Image
from services.transforms import TransformService


@pytest.fixture()
def random_image_path() -> str:
    return "../data/original/train/1.jpg"


def test_basic_transform(random_image_path):
    random_image = np.array(Image.open(random_image_path))

    input_image = random_image[:, :random_image.shape[1] // 2, :]
    target_image = random_image[:, random_image.shape[1] // 2:, :]

    assert input_image.shape == (
        600,
        600,
        3,
    ), f"Input image shape is incorrect: {input_image.shape}"
    assert target_image.shape == (
        600,
        600,
        3,
    ), f"Target image shape is incorrect: {target_image.shape}"

    transformed = TransformService.basic_transform(image_size=256)(
        image=input_image, mask=target_image)

    assert transformed["image"].shape == (256, 256, 3), (
        "Transform did not resize the image correctly for " "input image."
    )
    assert transformed["mask"].shape == (256, 256, 3), (
        "Transform did not resize the image correctly for " "target image"
    )
