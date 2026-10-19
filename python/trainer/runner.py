"""Line-delimited JSON pipe adapter between the client and trainer coordinator."""

import json
from pathlib import Path
from typing import TextIO

from .coordinator import TrainingCoordinator


PROTOCOL_VERSION = 1


def _send(output: TextIO, message: dict) -> None:
    output.write(json.dumps(message, separators=(",", ":")) + "\n")
    output.flush()


def _receive(input: TextIO) -> dict:
    line = input.readline()
    if not line:
        raise EOFError("client command pipe closed")
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("client command must be a JSON object")
    return message


def run(coordinator: TrainingCoordinator, input: TextIO, output: TextIO) -> None:
    """Request shards until the client finishes at a shard boundary."""
    try:
        _send(
            output,
            {
                "type": "ready",
                "protocol_version": PROTOCOL_VERSION,
            },
        )
        if coordinator.last_checkpointed_index >= 0:
            # The previous process may have exited before the client handled it.
            _send_checkpoint(output, coordinator)
        while True:
            _send(output, {"type": "shard_request", "index": coordinator.next_index})
            response = _receive(input)
            if response == {"type": "finish"}:
                previous = coordinator.last_checkpointed_index
                coordinator.checkpoint()
                if coordinator.last_checkpointed_index != previous:
                    _send_checkpoint(output, coordinator)
                _send(output, {"type": "stopped"})
                return
            if (
                set(response) != {"type", "path"}
                or response["type"] != "shard"
                or not isinstance(response["path"], str)
                or not Path(response["path"]).is_absolute()
            ):
                raise ValueError("expected shard response with an absolute path or finish")

            previous = coordinator.last_checkpointed_index
            index, rows = coordinator.train_next(Path(response["path"]))
            if coordinator.last_checkpointed_index != previous:
                _send_checkpoint(output, coordinator)
            _send(output, {"type": "shard_trained", "index": index, "rows": rows})
    except Exception as error:
        _send(output, {"type": "error", "message": str(error)})
        raise


def _send_checkpoint(output: TextIO, coordinator: TrainingCoordinator) -> None:
    _send(
        output,
        {
            "type": "checkpoint_ready",
            "path": str(coordinator.checkpoint_path),
            "last_shard_index": coordinator.last_checkpointed_index,
        },
    )
