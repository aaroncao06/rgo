import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from trainer.dummy_model import DummyModel
from trainer.checkpoint import save_checkpoint
from trainer.coordinator import TrainingCoordinator


def train_two_rows(model, optimizer, path, _base_batch_size, _effective_batch_size):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    (loss * (int(path.stem) + 1)).backward()
    optimizer.step()
    return 2


def new_coordinator(path, checkpoint_rows):
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return TrainingCoordinator(model, optimizer, path, 1, 8, checkpoint_rows)


class TrainingCoordinatorTest(unittest.TestCase):
    def test_restart_replays_only_work_after_last_checkpoint(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer.pt"

            with patch("trainer.coordinator.train_shard", side_effect=train_two_rows):
                first = new_coordinator(path, checkpoint_rows=4)
                self.assertEqual(first.next_index, 0)
                self.assertEqual(first.train_next(Path("0.npz")), (0, 2))
                self.assertEqual(first.last_checkpointed_index, -1)
                self.assertEqual(first.train_next(Path("1.npz")), (1, 2))
                self.assertEqual(first.last_checkpointed_index, 1)
                self.assertEqual(first.train_next(Path("2.npz")), (2, 2))
                self.assertEqual(first.last_checkpointed_index, 1)

                restarted = new_coordinator(path, checkpoint_rows=4)
                self.assertEqual(restarted.next_index, 2)
                self.assertEqual(restarted.train_next(Path("2.npz")), (2, 2))
                for expected, actual in zip(
                    first.model.parameters(), restarted.model.parameters()
                ):
                    torch.testing.assert_close(expected, actual)

                restarted.checkpoint()
                self.assertEqual(new_coordinator(path, 4).next_index, 3)

    def test_failed_training_cannot_advance_or_retry_dirty_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer.pt"
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
            self.assertEqual(new_coordinator(path, 1).next_index, 1)

    def test_checkpoint_error_after_replacement_recovers_new_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer.pt"
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
            self.assertEqual(new_coordinator(path, 1).next_index, 1)


if __name__ == "__main__":
    unittest.main()
