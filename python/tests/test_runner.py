import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_coordinator import new_coordinator, train_two_rows
from trainer.runner import run


def messages(output):
    return [json.loads(line) for line in output.getvalue().splitlines()]


class TrainerRunnerTest(unittest.TestCase):
    def test_train_then_finish_checkpoints_at_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trainer.pt"
            shard = Path(directory) / "0.npz"
            commands = io.StringIO(
                '\n'.join(
                    json.dumps(command)
                    for command in (
                        {"type": "shard", "path": str(shard)},
                        {"type": "finish"},
                    )
                ) + '\n'
            )
            output = io.StringIO()
            with patch("trainer.coordinator.train_shard", side_effect=train_two_rows):
                run(new_coordinator(checkpoint, checkpoint_rows=4), commands, output)

            self.assertEqual(
                messages(output),
                [
                    {"type": "ready", "protocol_version": 1},
                    {"type": "shard_request", "index": 0},
                    {"type": "shard_trained", "index": 0, "rows": 2},
                    {"type": "shard_request", "index": 1},
                    {"type": "checkpoint_ready", "path": str(checkpoint), "last_shard_index": 0},
                    {"type": "stopped"},
                ],
            )
            self.assertEqual(new_coordinator(checkpoint, 4).next_index, 1)

    def test_automatic_checkpoint_precedes_shard_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trainer.pt"
            shard = Path(directory) / "0.npz"
            commands = io.StringIO(
                json.dumps({"type": "shard", "path": str(shard)}) + '\n'
                + json.dumps({"type": "finish"}) + '\n'
            )
            output = io.StringIO()
            with patch("trainer.coordinator.train_shard", side_effect=train_two_rows):
                run(new_coordinator(checkpoint, checkpoint_rows=2), commands, output)

            self.assertEqual(
                [message["type"] for message in messages(output)],
                ["ready", "shard_request", "checkpoint_ready", "shard_trained", "shard_request", "stopped"],
            )
            self.assertEqual(new_coordinator(checkpoint, 2).next_index, 1)

    def test_restart_requests_next_checkpointed_index(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trainer.pt"
            output = io.StringIO()
            commands = io.StringIO(
                json.dumps({"type": "shard", "path": str(Path(directory) / "0.npz")}) + '\n'
                + json.dumps({"type": "finish"}) + '\n'
            )
            with patch("trainer.coordinator.train_shard", side_effect=train_two_rows):
                run(new_coordinator(checkpoint, 4), commands, output)

            restarted_output = io.StringIO()
            restarted_commands = io.StringIO(
                json.dumps({"type": "shard", "path": str(Path(directory) / "1.npz")}) + '\n'
                + json.dumps({"type": "finish"}) + '\n'
            )
            with patch("trainer.coordinator.train_shard", side_effect=train_two_rows):
                run(new_coordinator(checkpoint, 4), restarted_commands, restarted_output)
            self.assertEqual(messages(restarted_output)[0], {"type": "ready", "protocol_version": 1})
            self.assertEqual(
                messages(restarted_output)[1],
                {"type": "checkpoint_ready", "path": str(checkpoint), "last_shard_index": 0},
            )
            self.assertEqual(messages(restarted_output)[2], {"type": "shard_request", "index": 1})
            self.assertEqual(new_coordinator(checkpoint, 4).next_index, 2)

    def test_bad_shard_response_reports_error_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trainer.pt"
            commands = io.StringIO('{"type":"shard","path":"relative.npz"}\n')
            output = io.StringIO()
            with self.assertRaisesRegex(ValueError, "absolute path"):
                run(new_coordinator(checkpoint, 2), commands, output)
            self.assertEqual(messages(output)[-1]["type"], "error")
            self.assertFalse(checkpoint.exists())

    def test_finish_before_first_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trainer.pt"
            output = io.StringIO()
            run(new_coordinator(checkpoint, 2), io.StringIO('{"type":"finish"}\n'), output)
            self.assertEqual(
                [message["type"] for message in messages(output)],
                ["ready", "shard_request", "stopped"],
            )
            self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
