import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from trainer.dataset import ShardDataset
from trainer.shards import load_shard


class ShardTest(unittest.TestCase):
    def test_loads_packed_groups_without_expanding(self):
        arrays = {
            "format_version": np.array(1, dtype=np.uint32),
            "board_sizes": np.array([9, 13, 19], dtype=np.uint8),
        }
        for size in (9, 13, 19):
            area = size * size
            spatial = np.zeros((2, 3, (area + 7) // 8), dtype=np.uint8)
            spatial[0, 0, 0] = 0b00000101  # Cells 0 and 2.
            spatial[0, 2, -1] = 1 << ((area - 1) % 8)
            spatial[1, 1, 0] = 0b00000010  # Row 1, cell 1.
            ownership = np.zeros((2, (2 * area + 7) // 8), dtype=np.uint8)
            ownership[0, 0] = 0b10010010  # Cells 0..3: 2, 0, 1, 2.
            ownership[1, 0] = 1
            policy = np.zeros((2, area + 1), dtype=np.float16)
            policy[0, 0], policy[0, -1] = 0.3, 0.7
            policy[1, -1] = 1
            arrays.update({
                f"{size}/spatial": spatial,
                f"{size}/global": np.array([[7.5, 1], [-7.5, 0]], dtype=np.float32),
                f"{size}/policy": policy,
                f"{size}/value": np.array([[1, 5], [0, -5]], dtype=np.float32),
                f"{size}/ownership": ownership,
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, **arrays)
            groups = load_shard(path)
            dataset = ShardDataset(path, base_batch_size=1)

        self.assertEqual([group.board_size for group in groups], [9, 13, 19])
        for group in groups:
            size = group.board_size
            self.assertEqual(group.spatial.shape, (2, 3, (size * size + 7) // 8))
            self.assertEqual(group.spatial.dtype, np.uint8)
            self.assertEqual(group.spatial[0, 0, 0], 0b00000101)
            self.assertEqual(group.ownership.shape, (2, (2 * size * size + 7) // 8))
            self.assertEqual(group.ownership[0, 0], 0b10010010)
            self.assertEqual(group.global_features[0].tolist(), [7.5, 1])
            self.assertEqual(group.global_features[1].tolist(), [-7.5, 0])
            self.assertEqual(group.value[0].tolist(), [1, 5])
            self.assertEqual(group.value[1].tolist(), [0, -5])
            self.assertEqual(group.policy.dtype, np.float16)
            self.assertGreater(group.policy[0, -1], group.policy[0, 0])

        batches = [dataset[index] for index in range(len(dataset))]
        self.assertEqual([batch.board_size for batch in batches], [9, 13, 19, 19])
        for size in (9, 13, 19):
            batch = next(batch for batch in batches if batch.board_size == size)
            self.assertEqual(batch.spatial.dtype, torch.float32)
            self.assertEqual(batch.spatial[0, 0, 0, :4].tolist(), [1, 0, 1, 0])
            self.assertEqual(batch.spatial[0, 2, -1, -1], 1)
            self.assertEqual(batch.ownership[0, 0, :4].tolist(), [2, 0, 1, 2])
            self.assertEqual(batch.policy.dtype, torch.float32)
            torch.testing.assert_close(batch.policy.sum(dim=1), torch.ones(len(batch.policy)))


if __name__ == "__main__":
    unittest.main()
