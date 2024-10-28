from albumentations import Compose, Resize, HorizontalFlip, ColorJitter, Normalize
from albumentations.pytorch import ToTensorV2


class TransformService:
    @staticmethod
    def basic_transform(image_size):
        return Compose([
            Resize(width=image_size, height=image_size)
        ])

    @staticmethod
    def input_transform():
        return Compose([
            HorizontalFlip(p=0.3),
            ColorJitter(p=0.2),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2()
        ])

    @staticmethod
    def target_transform():
        return Compose([
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2()
        ])
