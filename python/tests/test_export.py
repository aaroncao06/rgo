import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch

from trainer.checkpoint import save_checkpoint
from model.dummy_model import DummyModel
from server.export import export_checkpoint


class ExportTest(unittest.TestCase):
    def test_checkpoint_exports_match_pytorch_for_all_board_and_batch_sizes(self):
        torch.manual_seed(12)
        model = DummyModel().eval()
        optimizer = torch.optim.Adam(model.parameters())
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in [-1, 7]:
                with torch.no_grad():
                    model.value.bias.add_(0.5)
                checkpoint = save_checkpoint(root, model, optimizer, index)
                output = root / f"{index + 1}.onnx"
                if index == -1:
                    # Exercise the subprocess interface the server will call.
                    result = subprocess.run(
                        [sys.executable, "-m", "server.export", str(checkpoint), str(output)],
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    export_checkpoint(checkpoint, output)
                graph = onnx.load(output)
                self.assertEqual(
                    {prop.key: prop.value for prop in graph.metadata_props},
                    {"rgo.io_version": "0"},
                )
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
                self.assertTrue(all(
                    port.type == "tensor(float)"
                    for port in session.get_inputs() + session.get_outputs()
                ))
                self.assertEqual(
                    {port.name: port.shape for port in session.get_inputs()},
                    {"spatial": ["batch", 3, "height", "width"], "global": ["batch", 2]},
                )
                self.assertEqual(
                    {port.name: port.shape for port in session.get_outputs()},
                    {"policy_logits": ["batch", "moves"], "value": ["batch", 3],
                     "ownership_logits": ["batch", 1, "height", "width"]},
                )
                for size in [9, 13, 19]:
                    for batch in [1, 3]:
                        with self.subTest(index=index, size=size, batch=batch):
                            spatial = rng.integers(0, 2, (batch, 3, size, size)).astype(np.float32)
                            global_features = rng.normal(size=(batch, 2)).astype(np.float32)
                            with torch.no_grad():
                                expected = model(torch.from_numpy(spatial), torch.from_numpy(global_features))
                            actual = session.run(None, {"spatial": spatial, "global": global_features})
                            for reference, exported in zip(expected, actual):
                                np.testing.assert_allclose(exported, reference.numpy(), rtol=1e-5, atol=1e-6)
                self.assertFalse(list(root.glob(".*.onnx.*")))

    def test_failed_export_leaves_no_final_or_temporary_file(self):
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_checkpoint(root, model, optimizer, -1)

            def fail_after_partial_write(_model, _inputs, file, **_kwargs):
                file.write(b"partial export")
                raise OSError("simulated export failure")

            with patch("server.export.torch.onnx.export", side_effect=fail_after_partial_write):
                with self.assertRaises(OSError):
                    export_checkpoint(checkpoint, root / "0.onnx")
            self.assertEqual(list(root.iterdir()), [checkpoint])

    def test_existing_model_cannot_be_overwritten(self):
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_checkpoint(root, model, optimizer, 0)
            output = root / "1.onnx"
            output.write_bytes(b"previous model")
            with self.assertRaises(FileExistsError):
                export_checkpoint(checkpoint, output)
            self.assertEqual(output.read_bytes(), b"previous model")
            self.assertEqual(set(root.iterdir()), {checkpoint, output})

    @unittest.skipUnless(os.name == "posix", "directory fsync is POSIX-only")
    def test_directory_sync_failure_reports_failure_but_keeps_the_final_model(self):
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_checkpoint(root, model, optimizer, -1)
            output = root / "0.onnx"
            with patch("server.export.os.fsync", side_effect=[None, OSError("sync failed")]):
                with self.assertRaises(OSError):
                    export_checkpoint(checkpoint, output)
            onnx.checker.check_model(onnx.load(output))
            self.assertEqual(set(root.iterdir()), {checkpoint, output})


if __name__ == "__main__":
    unittest.main()
