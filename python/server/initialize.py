"""Publish the initial weights for a new local run."""

import argparse
from pathlib import Path

from model.dummy_model import DummyModel
from trainer.checkpoint import save_checkpoint


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Create the initial RGO checkpoint")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    save_checkpoint(args.directory, DummyModel(), None, -1)


if __name__ == "__main__":
    main()
