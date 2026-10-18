import unittest

import torch

from dummy_model import DummyModel


class DummyModelTest(unittest.TestCase):
    def test_supported_board_sizes_and_gradients(self):
        model = DummyModel()
        for size in (9, 13, 19):
            with self.subTest(size=size):
                spatial = torch.rand(2, 3, size, size)
                global_features = torch.tensor([[7.5, 0.0], [-7.5, 1.0]])
                policy, value, ownership = model(spatial, global_features)
                self.assertEqual(tuple(policy.shape), (2, size * size + 1))
                self.assertEqual(tuple(value.shape), (2, 3))
                self.assertEqual(tuple(ownership.shape), (2, 1, size, size))
                (policy.sum() + value.sum() + ownership.sum()).backward()
                self.assertTrue(all(p.grad is not None for p in model.parameters()))
                model.zero_grad()


if __name__ == "__main__":
    unittest.main()
