"""Advance one trainer through indexed shards and durable checkpoint boundaries."""

from pathlib import Path

import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .train import train_shard


class TrainingCoordinator:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Path,
        base_batch_size: int,
        effective_batch_size: int,
        checkpoint_rows: int,
    ):
        if checkpoint_rows < 1:
            raise ValueError("checkpoint_rows must be positive")
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.base_batch_size = base_batch_size
        self.effective_batch_size = effective_batch_size
        self.checkpoint_rows = checkpoint_rows
        self.last_checkpointed_index = (
            load_checkpoint(checkpoint_path, model, optimizer)
            if checkpoint_path.exists()
            else -1
        )
        self.last_consumed_index = self.last_checkpointed_index
        self.rows_since_checkpoint = 0
        self._failed = False

    @property
    def next_index(self) -> int:
        return self.last_consumed_index + 1

    def train_next(self, path: Path) -> tuple[int, int]:
        """Train the next shard and checkpoint at the row threshold."""
        if self._failed:
            raise RuntimeError("reload the coordinator from its checkpoint after failure")
        index = self.next_index
        try:
            rows = train_shard(
                self.model,
                self.optimizer,
                path,
                self.base_batch_size,
                self.effective_batch_size,
            )
            if rows < 1:
                raise ValueError("training shard contains no rows")
            self.last_consumed_index = index
            self.rows_since_checkpoint += rows
            if self.rows_since_checkpoint >= self.checkpoint_rows:
                self.checkpoint()
            return index, rows
        except BaseException:
            # Optimizer state may have changed even if the shard did not finish.
            self._failed = True
            raise

    def checkpoint(self) -> None:
        """Persist completed work before a planned stop or trainer handoff."""
        if self._failed:
            raise RuntimeError("reload the coordinator from its checkpoint after failure")
        if self.last_consumed_index == self.last_checkpointed_index:
            return
        try:
            save_checkpoint(
                self.checkpoint_path,
                self.model,
                self.optimizer,
                self.last_consumed_index,
            )
        except BaseException:
            # Replacement may have succeeded before the error. Reopen to decide.
            self._failed = True
            raise
        self.last_checkpointed_index = self.last_consumed_index
        self.rows_since_checkpoint = 0
