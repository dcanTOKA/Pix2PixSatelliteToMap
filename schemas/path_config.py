from pydantic import BaseModel


class PathsConfig(BaseModel):
    root_dir: str
    train_dir: str
    val_dir: str
