import torch
import os
from . import config_
from torchvision.utils import save_image
from generator.models.model import Generator
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger('utils_logger')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def save_epoch_examples(gen: Generator, val_loader: DataLoader, epoch: int, folder: str):
    x, y = next(iter(val_loader))
    x, y = x.to(config_.training.device), y.to(config_.training.device)

    gen.eval()

    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5

        save_image(y_fake, os.path.join(folder, f"y_gen_{epoch}.png"))
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename):
    logger.info('=> Saving checkpoint...')

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, filename)


def load_checkpoint(model: torch.nn.Module, checkpoint_location: str, optimizer: torch.optim.Optimizer, lr: float):
    checkpoint = torch.load(checkpoint_location, map_location=config_.training.device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
