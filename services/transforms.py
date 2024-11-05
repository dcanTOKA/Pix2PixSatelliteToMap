from albumentations import Compose, Resize, HorizontalFlip, ColorJitter, Normalize
from albumentations.pytorch import ToTensorV2


class TransformService:
    @staticmethod
    def basic_transform(image_size):
        return Compose([
            Resize(width=image_size, height=image_size)
        ])

    @staticmethod
    def common_transform():
        return Compose([
            HorizontalFlip(p=0.3),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2()
        ], additional_targets={'mask': 'image'})
