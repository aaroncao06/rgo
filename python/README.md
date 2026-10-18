# Python trainer components

Use Python 3.9–3.12 and install the pinned NumPy and PyTorch versions in a
virtual environment. PyTorch 2.2.2 cannot convert arrays from NumPy 2.x.

From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r python/requirements.txt
PYTHONPATH=python:python/tests python -m unittest discover -s python/tests -p 'test_*.py'
```

`TrainingCoordinator` in `trainer/coordinator.py` connects indexed shard paths,
one-shard training, and local recovery checkpoints. Its shard-path callback is
the boundary for future Rust client integration; it is not a standalone trainer
command yet.
