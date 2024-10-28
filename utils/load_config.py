import yaml

from schemas.config import Config
from decouple import config


def load_config(path: str) -> Config:
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return Config(**data)


config_ = load_config(config('CONFIG_PATH'))
