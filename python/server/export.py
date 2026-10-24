"""Export accepted checkpoints using the temporary dummy model architecture."""

import argparse
import os
import tempfile
from pathlib import Path

import onnx
import torch

from model.dummy_model import DummyModel


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    """Publish an immutable V0 ONNX model in an existing output directory."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DummyModel()
    model.load_state_dict(state["model"])
    model.eval()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as file:
            temporary = Path(file.name)
            torch.onnx.export(
                model,
                (torch.zeros(1, 3, 19, 19), torch.zeros(1, 2)),
                file,
                opset_version=17,
                input_names=["spatial", "global"],
                output_names=["policy_logits", "value", "ownership_logits"],
                dynamic_axes={
                    "spatial": {0: "batch", 2: "height", 3: "width"},
                    "global": {0: "batch"},
                    "policy_logits": {0: "batch", 1: "moves"},
                    "value": {0: "batch"},
                    "ownership_logits": {0: "batch", 2: "height", 3: "width"},
                },
            )
            file.seek(0)
            graph = onnx.load(file)
            onnx.helper.set_model_props(graph, {"rgo.io_version": "0"})
            file.seek(0)
            file.truncate()
            onnx.save(graph, file)
            file.flush()
            os.fsync(file.fileno())
        # Like checkpoints, publish without overwriting an immutable final file.
        os.link(temporary, output)
        if os.name == "posix":
            handle = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export an accepted RGO checkpoint to ONNX")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
