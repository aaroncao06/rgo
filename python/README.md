# Python trainer components

Use [uv](https://docs.astral.sh/uv/) from the repository root. `pyproject.toml`
declares Python 3.9–3.12 and pins NumPy 1.26.4 with PyTorch 2.2.2; `uv.lock`
records their resolved dependencies. The default local Python is 3.12.
PyTorch 2.2.2 cannot convert arrays from NumPy 2.x.

From the repository root:

```sh
uv sync --locked
PYTHONPATH=python:python/tests uv run --locked python -m unittest discover -s python/tests -p 'test_*.py'
```

`TrainingCoordinator` in `trainer/coordinator.py` connects indexed shard paths,
one-shard training, and local recovery checkpoints. `trainer.runner.run` adapts
it to newline-delimited JSON streams: it reports `ready`, requests the next
shard index, accepts either an absolute local path or
`finish`, and reports training/checkpoint completion. An existing
checkpoint is announced again at startup so interrupted acceptance can be
retried. The client must handle each `checkpoint_ready` before replying with
another shard path because the coordinator reuses one checkpoint path.

`PYTHONPATH=python uv run --locked python -m trainer.main configs/trainer.toml` runs the
client-supervised trainer pipe worker. The TOML specifies a model, device,
learning rate, checkpoint path, and batch sizes. `build_trainer(config)`
currently constructs the temporary `DummyModel` and Adam optimizer; there is
no importable factory name in the TOML. The real ViT model and Rust client
launcher are still to be implemented.
