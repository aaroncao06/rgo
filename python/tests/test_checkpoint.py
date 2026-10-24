import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from model.dummy_model import DummyModel
from trainer.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint


def optimizer_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    spatial = torch.ones(2, 3, 9, 9)
    global_features = torch.tensor([[7.5, 0.0], [-7.5, 1.0]])
    policy, value, ownership = model(spatial, global_features)
    (policy.square().mean() + value.square().mean() + ownership.square().mean()).backward()
    optimizer.step()


class CheckpointTest(unittest.TestCase):
    def test_restores_model_optimizer_and_completed_shard(self):
        torch.manual_seed(0)
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer_step(model, optimizer)

        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint_path(Path(directory), 7)
            save_checkpoint(Path(directory), model, optimizer, last_shard_index=7)

            resumed = DummyModel()
            resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.001)
            self.assertEqual(load_checkpoint(path, resumed, resumed_optimizer), 7)

        optimizer_step(model, optimizer)
        optimizer_step(resumed, resumed_optimizer)
        for original, restored in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(original, restored)

    def test_failed_write_preserves_previous_checkpoint(self):
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint_path(Path(directory), 7)
            save_checkpoint(Path(directory), model, optimizer, last_shard_index=7)

            def fail_after_partial_write(_state, file):
                file.write(b"incomplete")
                raise OSError("simulated write failure")

            with patch("trainer.checkpoint.torch.save", side_effect=fail_after_partial_write):
                with self.assertRaises(OSError):
                    save_checkpoint(Path(directory), model, optimizer, last_shard_index=8)

            self.assertEqual(load_checkpoint(path, model, optimizer), 7)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_indexed_checkpoint_cannot_be_overwritten(self):
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            first = save_checkpoint(Path(directory), model, optimizer, 0)
            with self.assertRaises(FileExistsError):
                save_checkpoint(Path(directory), model, optimizer, 0)
            self.assertEqual(list(Path(directory).iterdir()), [first])


if __name__ == "__main__":
    unittest.main()
