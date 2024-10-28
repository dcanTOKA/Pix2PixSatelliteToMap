from pydantic import BaseModel


class ModelConfig(BaseModel):
    save_model: bool
    load_model: bool
    checkpoint_disc: str
    checkpoint_gen: str
    l1_lambda: int
