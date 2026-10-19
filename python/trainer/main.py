"""TOML entry point for the client-supervised trainer process."""

import argparse
import math
import sys
from pathlib import Path

import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9–3.10
    import tomli as tomllib

from .coordinator import TrainingCoordinator
from .dummy_model import DummyModel
from .runner import run


def parse_device(value: str) -> torch.device:
    if not isinstance(value, str):
        raise ValueError("device must be a PyTorch device string")
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid device: {value}") from error
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device must use cpu, cuda, or mps")
    return device


def build_trainer(config: dict) -> TrainingCoordinator:
    """Construct the temporary dummy trainer from local process settings."""
    fields = {
        "model",
        "device",
        "learning_rate",
        "checkpoint_path",
        "base_batch_size",
        "effective_batch_size",
        "checkpoint_rows",
    }
    if set(config) != fields:
        raise ValueError(
            f"trainer config fields: missing {sorted(fields - set(config))}, "
            f"unknown {sorted(set(config) - fields)}"
        )
    if config["model"] != "dummy":
        raise ValueError("only the temporary dummy model is implemented")
    device = parse_device(config["device"])
    rate = config["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    for key in ("base_batch_size", "effective_batch_size", "checkpoint_rows"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    size = config["effective_batch_size"]
    if size & (size - 1):
        raise ValueError("effective_batch_size must be a power of two")
    if not isinstance(config["checkpoint_path"], str) or not config["checkpoint_path"]:
        raise ValueError("checkpoint_path must be a nonempty string")

    checkpoint = Path(config["checkpoint_path"]).absolute()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = DummyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    return TrainingCoordinator(
        model,
        optimizer,
        checkpoint,
        config["base_batch_size"],
        size,
        config["checkpoint_rows"],
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run an RGO trainer pipe worker")
    parser.add_argument("config", type=Path, help="trainer TOML configuration")
    args = parser.parse_args(argv)
    with args.config.open("rb") as file:
        config = tomllib.load(file)
    run(build_trainer(config), sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
