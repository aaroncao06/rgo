"""Save and restore training state at a completed shard boundary."""

import os
import tempfile
from pathlib import Path

import torch
from torch import nn


def checkpoint_path(directory: Path, last_shard_index: int) -> Path:
    if last_shard_index < -1:
        raise ValueError("checkpoint shard index must be at least -1")
    return directory / f"{last_shard_index}.pt"


def save_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    last_shard_index: int,
) -> Path:
    path = checkpoint_path(directory, last_shard_index)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "last_shard_index": last_shard_index,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temporary = Path(file.name)
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path)
        if os.name == "posix":
            handle = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    state = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["last_shard_index"]
