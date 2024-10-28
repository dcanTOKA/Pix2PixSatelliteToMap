from pydantic import BaseModel

from schemas import ModelConfig
from schemas import PathsConfig
from schemas import TrainingConfig


class Config(BaseModel):
    training: TrainingConfig
    paths: PathsConfig
    model: ModelConfig
