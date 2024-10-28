from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from schemas.config import Config
from services import TransformService
from utils.load_config import config_


class SatelliteMapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.config: Config = config_

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        image_file = self.list_files[index]
        image_path = os.path.join(self.root_dir, image_file)
        image = np.array(Image.open(image_path))
        input_image = image[:, : image.shape[1] // 2, :]
        target_image = image[:, image.shape[1] // 2 :, :]

        transformed = TransformService.basic_transform(
            image_size=self.config.training.image_size
        )(image=input_image, mask=target_image)

        transformed_input_image = TransformService.input_transform()(
            image=transformed["image"]
        )
        transformed_target_image = TransformService.target_transform()(
            image=transformed["mask"]
        )

        return transformed_input_image["image"], transformed_target_image["image"]


if __name__ == "__main__":
    dataset = SatelliteMapDataset("../data/original/train")

    input_, target_ = dataset[0]

    plt.imshow(input_)
    print(input_.shape)
    plt.show()
