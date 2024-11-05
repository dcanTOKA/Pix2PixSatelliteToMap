import os.path

import torch.cuda.amp
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from discriminator.models.model import Discriminator
from generator.models.model import Generator
from utils.load_config import config_
from services.dataset import SatelliteMapDataset
from utils.training_util import save_checkpoint, load_checkpoint, save_epoch_examples


class TrainService:
    def __init__(self):
        self.generator = Generator(in_channels=config_.training.channels_img).to(config_.training.device)
        self.discriminator = Discriminator(in_channels=config_.training.channels_img).to(config_.training.device)

        self.generator_opt = AdamW(
            self.generator.parameters(),
            lr=config_.training.learning_rate,
            betas=(0.5, 0.999),
        )
        self.discriminator_opt = AdamW(
            self.discriminator.parameters(),
            lr=config_.training.learning_rate,
            betas=(0.5, 0.999),
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        self.gen_losses = []
        self.disc_losses = []

        if config_.model.load_model:
            load_checkpoint(
                self.generator,
                config_.model.checkpoint_gen,
                self.generator_opt,
                lr=config_.training.learning_rate,
            )
            load_checkpoint(
                self.discriminator,
                config_.model.checkpoint_disc,
                self.discriminator_opt,
                lr=config_.training.learning_rate,
            )

        self.train_dataset = SatelliteMapDataset(os.path.join(config_.paths.root_dir, config_.paths.train_dir))
        self.val_dataset = SatelliteMapDataset(os.path.join(config_.paths.root_dir, config_.paths.val_dir))

        self.train_dataloader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=config_.training.batch_size,
            shuffle=True,
            num_workers=config_.training.num_worker,
            drop_last=True
        )
        self.val_dataloader: DataLoader = DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False
        )

        self.generator_scaler = torch.amp.GradScaler(config_.training.device)
        self.discriminator_scaler = torch.amp.GradScaler(config_.training.device)

    def train_epoch(self, epoch):
        loop = tqdm(self.train_dataloader, leave=True)

        for idx, (x, y) in enumerate(loop):
            x: torch.Tensor = x.to(config_.training.device)
            y: torch.Tensor = y.to(config_.training.device)

            self.discriminator_opt.zero_grad()

            with torch.amp.autocast(config_.training.device):
                y_fake = self.generator(x)
                disc_fake = self.discriminator(x, y_fake.detach())
                disc_fake_loss = self.bce_loss(disc_fake, torch.zeros_like(disc_fake))

                disc_real = self.discriminator(x, y)
                disc_real_loss = self.bce_loss(disc_real, torch.ones_like(disc_real))

                disc_loss = (disc_real_loss + disc_fake_loss) / 2
                self.disc_losses.append(disc_loss.item())

            self.discriminator_scaler.scale(disc_loss).backward()
            self.discriminator_scaler.step(self.discriminator_opt)
            self.discriminator_scaler.update()

            self.generator_opt.zero_grad()

            with torch.amp.autocast(config_.training.device):
                disc_pred = self.discriminator(x, y_fake)
                gen_fake_loss = self.bce_loss(disc_pred, torch.ones_like(disc_pred))

                l1_loss = self.l1_loss(y_fake, y) * config_.model.l1_lambda
                gen_loss = l1_loss + gen_fake_loss
                self.gen_losses.append(gen_loss.item())

            self.generator_scaler.scale(gen_loss).backward()
            self.generator_scaler.step(self.generator_opt)
            self.generator_scaler.update()

            if idx % 2 == 0:
                loop.set_postfix(
                    Epoch=epoch,
                    D_real=torch.sigmoid(disc_real).mean().item(),
                    D_fake=torch.sigmoid(disc_pred).mean().item(),
                )

    def train(self):
        for epoch in range(config_.training.num_epochs):
            self.train_epoch(epoch)
            if config_.model.save_model and epoch % 5 == 0:
                save_checkpoint(self.generator, self.generator_opt, filename=config_.model.checkpoint_gen)
                save_checkpoint(self.discriminator, self.discriminator_opt, filename=config_.model.checkpoint_disc)

            save_epoch_examples(self.generator, self.val_dataloader, epoch, folder="../evaluation")


if __name__ == "__main__":
    train_service = TrainService()
    train_service.train()
