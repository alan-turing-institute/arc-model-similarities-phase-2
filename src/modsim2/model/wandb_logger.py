# The code below taken from pytorch_lightning 1.9.3 and adapted
# to the project's version of 1.8.3.post1 to avoid updating
# and breaking everything - see issue #37

from pathlib import Path
from typing import Any, Optional, Union

import wandb
from lightning_lite.utilities.types import _PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import _WANDB_GREATER_EQUAL_0_10_22, WandbLogger
from pytorch_lightning.utilities.logger import _scan_checkpoints
from torch import Tensor
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run


class MS2WandbLogger(WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: str = "lightning_logs",
        log_model: Union[str, bool] = False,
        experiment: Union[Run, RunDisabled, None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            save_dir,
            version,
            offline,
            dir,
            id,
            anonymous,
            project,
            log_model,
            experiment,
            prefix,
            **kwargs,
        )

        self._checkpoint_name = checkpoint_name

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = (
                {
                    "score": s.item() if isinstance(s, Tensor) else s,
                    "original_filename": Path(p).name,
                    checkpoint_callback.__class__.__name__: {
                        k: getattr(checkpoint_callback, k)
                        for k in [
                            "monitor",
                            "mode",
                            "save_last",
                            "save_top_k",
                            "save_weights_only",
                            "_every_n_train_steps",
                        ]
                        # ensure it does not break if `ModelCheckpoint`
                        # args change
                        if hasattr(checkpoint_callback, k)
                    },
                }
                if _WANDB_GREATER_EQUAL_0_10_22
                else None
            )
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"
            artifact = wandb.Artifact(
                name=self._checkpoint_name, type="model", metadata=metadata
            )
            artifact.add_file(p, name="model.ckpt")
            self.experiment.log_artifact(artifact, aliases=[tag])
            # remember logged models - timestamp needed in case filename
            # didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
