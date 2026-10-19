import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.dummy_model import DummyModel
from trainer.dataset import ShardDataset
from trainer.shards import PackedBoardSamples
from trainer.train import train_shard


def group(size, ids):
    n = len(ids)
    global_features = np.zeros((n, 2), dtype=np.float32)
    global_features[:, 0] = ids
    policy = np.zeros((n, size * size + 1), dtype=np.float16)
    policy[:, 0] = 1
    policy[:, -1] = 3
    spatial = np.zeros((n, 3, (size * size + 7) // 8), dtype=np.uint8)
    spatial[:, 0, 0] = 0b00000101
    ownership = np.zeros((n, (2 * size * size + 7) // 8), dtype=np.uint8)
    ownership[:, 0] = 0b10010010
    return PackedBoardSamples(
        size,
        spatial,
        global_features,
        policy,
        np.zeros((n, 2), dtype=np.float32),
        ownership,
    )


class ShardDatasetTest(unittest.TestCase):
    def test_batches_one_shard_by_board_size_without_losing_rows(self):
        groups = [
            group(9, list(range(9))),
            group(13, [100, 101, 102]),
            group(19, [200]),
        ]
        with patch("trainer.dataset.load_shard", return_value=groups) as load:
            dataset = ShardDataset(
                Path("sample.npz"), base_batch_size=1, effective_batch_size=128
            )
            self.assertEqual(dataset.groups[0].spatial.dtype, np.uint8)
            batches = list(DataLoader(dataset, batch_size=None))

        load.assert_called_once_with(Path("sample.npz"))
        self.assertEqual(
            [(batch.board_size, batch.global_features[:, 0].tolist()) for batch in batches],
            [
                (9, [0, 1, 2, 3]),
                (9, [4, 5, 6, 7]),
                (9, [8]),
                (13, [100, 101]),
                (13, [102]),
                (19, [200]),
            ],
        )
        for batch in batches:
            self.assertEqual(batch.spatial.shape[2:], (batch.board_size, batch.board_size))
            self.assertEqual(len(batch.policy), len(batch.global_features))
            self.assertEqual(batch.spatial.dtype, torch.float32)
            self.assertEqual(batch.ownership.dtype, torch.int64)
            self.assertEqual(batch.spatial[0, 0, 0, :4].tolist(), [1, 0, 1, 0])
            self.assertEqual(batch.ownership[0, 0, :4].tolist(), [2, 0, 1, 2])
            torch.testing.assert_close(batch.policy.sum(dim=1), torch.ones(len(batch.policy)))
            torch.testing.assert_close(
                batch.policy[:, [0, -1]],
                torch.tensor([[0.25, 0.75]] * len(batch.policy)),
            )
        self.assertEqual(dataset.groups[0].policy.dtype, np.float16)
        self.assertEqual(dataset.groups[0].policy[0, [0, -1]].tolist(), [1, 3])

        batch = batches[0]
        with torch.no_grad():
            policy, value, ownership = DummyModel()(batch.spatial, batch.global_features)
        self.assertEqual(policy.shape, batch.policy.shape)
        self.assertEqual(value.shape, (len(batch.policy), 3))
        self.assertEqual(ownership.shape, (len(batch.policy), 1, 9, 9))

    def test_rejects_empty_base_batch_size(self):
        with self.assertRaises(ValueError):
            ShardDataset(Path("sample.npz"), base_batch_size=0, effective_batch_size=1)

    def test_rejects_empty_effective_batch_size(self):
        with self.assertRaises(ValueError):
            ShardDataset(Path("sample.npz"), base_batch_size=1, effective_batch_size=0)

    def test_rejects_non_power_of_two_effective_batch_size(self):
        with self.assertRaises(ValueError):
            ShardDataset(Path("sample.npz"), base_batch_size=1, effective_batch_size=3)

    def test_training_updates_at_fixed_row_boundaries_across_board_sizes(self):
        groups = [
            group(9, list(range(5))),
            group(13, list(range(5, 9))),
            group(19, [9, 10]),
        ]
        torch.manual_seed(0)
        first = DummyModel()
        second = deepcopy(first)
        before = {
            name: parameter.detach().clone()
            for name, parameter in first.named_parameters()
        }

        with patch("trainer.dataset.load_shard", return_value=groups):
            dataset = ShardDataset(Path("sample.npz"), 1, 8)
            self.assertEqual(
                [stop - start for _, start, stop in dataset.batches],
                [4, 1, 2, 1, 1, 1, 1],
            )
            for model, base_batch_size in ((first, 1), (second, 2)):
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    self.assertEqual(
                        train_shard(model, optimizer, Path("sample.npz"), base_batch_size, 8),
                        11,
                    )
                self.assertEqual(step.call_count, 2)

        for name, parameter in first.named_parameters():
            with self.subTest(name=name):
                self.assertFalse(torch.equal(parameter, before[name]))
                torch.testing.assert_close(parameter, dict(second.named_parameters())[name])

    def test_partial_final_update_uses_its_actual_row_count(self):
        torch.manual_seed(1)
        first = DummyModel()
        second = deepcopy(first)
        with patch("trainer.dataset.load_shard", return_value=[group(9, [0, 1])]):
            for model, effective_batch_size in ((first, 2), (second, 16)):
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                train_shard(model, optimizer, Path("sample.npz"), 1, effective_batch_size)

        for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(first_parameter, second_parameter)


if __name__ == "__main__":
    unittest.main()
