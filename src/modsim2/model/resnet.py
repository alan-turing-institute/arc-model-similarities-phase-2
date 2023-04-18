# Modified from mod sim phase 1 code

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torchmetrics import Accuracy


def create_model(num_classes=10, channels=3):
    """
    Function to create a resnet18 model for CIAFAR-10.

    Args:
        num_classes: Number of classes in the dataset
        channels: Number of channels in the image

    Returns: ResNet18 model
    """
    # pretrained=False to avoid messing up our modification to conv1 below
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

    # ResNet is designed for ImageNet which has high res images;
    # use a smaller kernel size for CIFAR-10.
    model.conv1 = torch.nn.Conv2d(
        channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = torch.nn.Identity()
    return model


class ResnetModel(pl.LightningModule):
    """
    ResNet18 model lightning module for use in project
    """

    def __init__(
        self,
        num_classes=10,
        lr=0.05,
        weight_decay=0.0005,
        momentum=0.9,
        batch_size=32,
        train_size=50000,
        channels=3,
    ):
        super().__init__()
        self.model = create_model(num_classes=num_classes, channels=channels)
        self.num_classes = num_classes
        if num_classes == 2:
            self.task = "binary"
        if num_classes > 2:
            self.task = "multiclass"
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = self.momentum
        self.steps_per_epoch = train_size // batch_size
        self.optimizer = None

    def forward(self, x):
        out = self.model(x)
        return torch.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1).cpu()
        y = y.cpu()
        accuracy = Accuracy(task=self.task, num_classes=self.num_classes, top_k=1)
        acc = accuracy(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        softmax = torch.nn.Softmax(dim=1)
        logits = self.model(batch)
        return softmax(logits)

    def on_train_epoch_end(self):
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
        )

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
