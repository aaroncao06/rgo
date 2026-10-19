import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from trainer.main import parse_device


class TrainerMainTest(unittest.TestCase):
    def test_accepts_explicit_cuda_index(self):
        self.assertEqual(parse_device("cuda:0"), torch.device("cuda:0"))
        self.assertEqual(parse_device("cuda:1"), torch.device("cuda:1"))
        with self.assertRaisesRegex(ValueError, "cpu, cuda, or mps"):
            parse_device("meta")

    def test_dummy_trainer_process_trains_and_checkpoints_one_shard(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            config = Path(directory) / "trainer.toml"
            config.write_text(
                'model = "dummy"\n'
                'device = "cpu"\n'
                'learning_rate = 0.001\n'
                f'checkpoint_path = {json.dumps(str(checkpoint))}\n'
                'base_batch_size = 1\n'
                'effective_batch_size = 1\n'
                'checkpoint_rows = 1\n'
            )
            shard = Path(directory) / "0.npz"
            area = 9 * 9
            policy = np.zeros((1, area + 1), dtype=np.float16)
            policy[0, -1] = 1
            np.savez(
                shard,
                format_version=np.array(1, dtype=np.uint32),
                board_sizes=np.array([9], dtype=np.uint8),
                **{
                    "9/spatial": np.zeros((1, 3, (area + 7) // 8), dtype=np.uint8),
                    "9/global": np.zeros((1, 2), dtype=np.float32),
                    "9/policy": policy,
                    "9/value": np.array([[1, 0]], dtype=np.float32),
                    "9/ownership": np.zeros((1, (2 * area + 7) // 8), dtype=np.uint8),
                },
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(root / "python"), environment.get("PYTHONPATH")))
            )
            result = subprocess.run(
                [sys.executable, "-m", "trainer.main", str(config)],
                input=(
                    json.dumps({"type": "shard", "path": str(shard)}) + "\n"
                    + '{"type":"finish"}\n'
                ),
                capture_output=True,
                text=True,
                cwd=root,
                env=environment,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(
                [event["type"] for event in events],
                ["ready", "shard_request", "checkpoint_ready", "shard_trained", "shard_request", "stopped"],
            )
            self.assertEqual(events[1]["index"], 0)
            self.assertEqual(events[2]["last_shard_index"], 0)
            self.assertEqual(events[3]["rows"], 1)
            self.assertEqual(events[4]["index"], 1)
            self.assertTrue(checkpoint.is_file())

            restarted = subprocess.run(
                [sys.executable, "-m", "trainer.main", str(config)],
                input='{"type":"finish"}\n',
                capture_output=True,
                text=True,
                cwd=root,
                env=environment,
                timeout=30,
            )
            self.assertEqual(restarted.returncode, 0, restarted.stderr)
            recovered = [json.loads(line) for line in restarted.stdout.splitlines()]
            self.assertEqual(
                [event["type"] for event in recovered],
                ["ready", "checkpoint_ready", "shard_request", "stopped"],
            )
            self.assertEqual(recovered[2]["index"], 1)
            self.assertEqual(recovered[1]["last_shard_index"], 0)


if __name__ == "__main__":
    unittest.main()
