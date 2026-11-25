import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from model.dummy_model import DummyModel
from trainer.checkpoint import checkpoint_path, save_checkpoint
from trainer.coordinator import TrainingCoordinator


def train_two_rows(model, optimizer, path, _base_batch_size, _effective_batch_size):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    (loss * (int(path.stem) + 1)).backward()
    optimizer.step()
    return 2


def new_coordinator(directory, checkpoint_rows, resume_path=None):
    model = DummyModel()
    if resume_path is None:
        server = directory / "server"
        server.mkdir()
        resume_path = save_checkpoint(server, model, None, -1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return TrainingCoordinator(model, optimizer, directory, resume_path, 1, 8, checkpoint_rows)


class TrainingCoordinatorTest(unittest.TestCase):
    def test_initial_checkpoint_uses_server_weights_and_trainer_optimizer_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            server_model = DummyModel()
            server = path / "server"
            server.mkdir()
            initial = save_checkpoint(server, server_model, None, -1)
            resumed = new_coordinator(path, checkpoint_rows=4, resume_path=initial)
            self.assertEqual(resumed.next_index, 0)
            self.assertEqual(resumed.optimizer.param_groups[0]["lr"], 0.001)
            self.assertFalse(resumed.optimizer.state)
            self.assertFalse(checkpoint_path(path, -1).exists())
            for expected, actual in zip(server_model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_restart_replays_only_work_after_last_checkpoint(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)

            with patch("trainer.coordinator.train_shard", side_effect=train_two_rows):
                first = new_coordinator(path, checkpoint_rows=4)
                self.assertEqual(first.next_index, 0)
                first.train_next(Path("0.npz"))
                self.assertEqual(first.next_index, 1)
                self.assertEqual(first.last_checkpointed_index, -1)
                first.train_next(Path("1.npz"))
                self.assertEqual(first.next_index, 2)
                self.assertEqual(first.last_checkpointed_index, 1)
                first.train_next(Path("2.npz"))
                self.assertEqual(first.next_index, 3)
                self.assertEqual(first.last_checkpointed_index, 1)

                restarted = new_coordinator(path, checkpoint_rows=4, resume_path=checkpoint_path(path, 1))
                self.assertEqual(restarted.next_index, 2)
                restarted.train_next(Path("2.npz"))
                self.assertEqual(restarted.next_index, 3)
                for expected, actual in zip(
                    first.model.parameters(), restarted.model.parameters()
                ):
                    torch.testing.assert_close(expected, actual)

                restarted.checkpoint()
                self.assertEqual(new_coordinator(path, 4, checkpoint_path(path, 2)).next_index, 3)

    def test_failed_training_cannot_advance_or_retry_dirty_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = new_coordinator(path, checkpoint_rows=1)
            with patch("trainer.coordinator.train_shard", side_effect=train_two_rows):
                first.train_next(Path("0.npz"))

            def partial_training(model, optimizer, shard, base, effective):
                train_two_rows(model, optimizer, shard, base, effective)
                raise OSError("failed after optimizer step")

            with patch("trainer.coordinator.train_shard", side_effect=partial_training):
                with self.assertRaisesRegex(OSError, "failed after optimizer step"):
                    first.train_next(Path("1.npz"))
            with self.assertRaisesRegex(RuntimeError, "reload the coordinator"):
                first.train_next(Path("1.npz"))
            self.assertEqual(new_coordinator(path, 1, checkpoint_path(path, 0)).next_index, 1)

    def test_checkpoint_error_after_replacement_recovers_new_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = new_coordinator(path, checkpoint_rows=1)

            def save_then_fail(*args):
                save_checkpoint(*args)
                raise OSError("failed after replacement")

            with patch("trainer.coordinator.train_shard", side_effect=train_two_rows):
                with patch("trainer.coordinator.save_checkpoint", side_effect=save_then_fail):
                    with self.assertRaisesRegex(OSError, "failed after replacement"):
                        first.train_next(Path("0.npz"))
            with self.assertRaisesRegex(RuntimeError, "reload the coordinator"):
                first.checkpoint()
            self.assertEqual(new_coordinator(path, 1, checkpoint_path(path, 0)).next_index, 1)


if __name__ == "__main__":
    unittest.main()
