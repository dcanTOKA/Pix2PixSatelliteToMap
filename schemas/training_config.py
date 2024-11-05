from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int
    image_size: int
    channels_img: int
    num_worker: int
    device: str = Field(default="cuda")
