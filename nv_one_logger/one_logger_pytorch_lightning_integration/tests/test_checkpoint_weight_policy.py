# SPDX-License-Identifier: Apache-2.0
"""Checkpoint forwarding must accept modern Lightning's optional policy."""
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import pytest

from nv_one_logger.training_telemetry.integration.pytorch_lightning import OneLoggerPTLTrainer, Trainer


@pytest.mark.parametrize("weights_only", [False, True, None])
def test_checkpoint_weight_policy_is_forwarded(weights_only):
    trainer = OneLoggerPTLTrainer.__new__(OneLoggerPTLTrainer)
    callback = trainer._nv_one_logger_callback = Mock()
    path = Path("checkpoint.ckpt")
    with patch.object(Trainer, "global_step", new_callable=PropertyMock, return_value=7), patch.object(
        Trainer, "save_checkpoint"
    ) as save:
        trainer.save_checkpoint(path, weights_only=weights_only)
    save.assert_called_once_with(path, weights_only, None)
    callback.on_save_checkpoint_start.assert_called_once_with(global_step=7)
    callback.on_save_checkpoint_success.assert_called_once_with(global_step=7)
    callback.on_save_checkpoint_end.assert_called_once_with()
