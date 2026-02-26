import os
import numpy as np
import json
import torch
from torch import nn
from torch.utils import data
from generator_core import TypedTimer


class Trainer:

    def __init__(
            self,

            # The model to be trained
            model: nn.Module,

            # The dataloader to be used for training
            train_dataloader: data.DataLoader,

            # The loss function and optimizer to be used for training
            criterion=nn.CrossEntropyLoss(ignore_index=0),
            optimizer=torch.optim.Adam,

            # How many times the dataset is trained through
            epochs: int = 5,
            # Override the epochs param and only use a fraction of the whole dataset
            epoch_fraction: float = None,

            # Optionally provide a validation dataloader
            val_dataloader: data.DataLoader = None,

            # How many steps between each saved checkpoint
            checkpoint_frequency: int = None,

            lr_scheduler=None,
            device='cpu',

            model_dir=None,
            model_name=None,

            model_outputs_adaptor=lambda x: x,
            model_train_step=lambda model, x, y: model(x),
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters())
        self.epochs = epochs
        self.epoch_fraction = epoch_fraction
        self.val_dataloader = val_dataloader
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir if model_dir else os.path.join('temp', 'checkpoints')
        self.model_name = model_name if model_name else model.__class__.__name__
        self.model_outputs_adaptor = model_outputs_adaptor
        self.model_train_step = model_train_step

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

        self.timer = TypedTimer(self.model_name)

    def to(self, device):
        self.device = device
        self.model.to(device)

    def train(self):
        for epoch in range(self.epochs):
            self._train_step()
            self._validate_step()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )

            if self.lr_scheduler: self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_step(self):
        self.timer.start("_train_step")

        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            self.timer.start("batch")
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model_train_step(self.model, inputs, labels)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())
            self.timer.end("batch")

            if self.epoch_fraction and i > len(self.train_dataloader) * self.epoch_fraction: break

            if self.epoch_fraction and self.timer.drag("_train_step", 1):
                self._validate_step()
                self.model.train()

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)
        self.timer.end("_train_step")

    def _validate_step(self):
        if not self.val_dataloader: return
        self.timer.start("_validate_step")

        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                self.timer.start("val_batch")
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())
                self.timer.end("val_batch")

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)
        self.timer.end("_validate_step")

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}_{}.pt".format(self.model_name, str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, f"model_{self.model_name}.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, f"loss_{self.model_name}.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
