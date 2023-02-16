import os
import shutil

import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from modsim2.data.loader import DMPair
from modsim2.model.resnet import ResnetModel

LOG_DIR = "tmp/test_logs/"


@pytest.fixture(autouse=True)
def setup_teardown():
    os.makedirs(LOG_DIR, exist_ok=False)
    yield
    # teardown
    shutil.rmtree(LOG_DIR)


def test_model_runs():
    # Set up trainer
    dmpair = DMPair(val_split=0.2)
    model = ResnetModel(lr=0.05)
    trainer = Trainer(
        max_epochs=1,
        max_steps=2,
        devices=None,
        logger=CSVLogger(save_dir=LOG_DIR),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )

    # Make sure that it runs
    try:
        trainer.fit(model, dmpair.A)
    except Exception as exc:
        assert False, f"trainer.fit raised an exception: {exc}"
