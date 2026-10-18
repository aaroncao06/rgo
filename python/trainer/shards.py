"""Decode the server's version-1 sampled NPZ shards for training."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BoardSamples:
    board_size: int
    spatial: np.ndarray
    global_features: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    ownership: np.ndarray


def load_shard(path: Path) -> list[BoardSamples]:
    groups = []
    with np.load(path, allow_pickle=False) as shard:
        if shard["format_version"].item() != 1:
            raise ValueError("unsupported shard format version")
        for board_size in shard["board_sizes"]:
            size = int(board_size)
            if size not in (9, 13, 19):
                raise ValueError(f"unsupported board size: {size}")
            area = size * size
            prefix = f"{size}/"
            packed_spatial = shard[prefix + "spatial"]
            packed_ownership = shard[prefix + "ownership"]
            global_features = shard[prefix + "global"]
            policy = shard[prefix + "policy"].astype(np.float32)
            value = shard[prefix + "value"]
            rows = packed_spatial.shape[0]
            if (
                packed_spatial.shape != (rows, 3, (area + 7) // 8)
                or packed_ownership.shape != (rows, (2 * area + 7) // 8)
                or global_features.shape != (rows, 2)
                or policy.shape != (rows, area + 1)
                or value.shape != (rows, 2)
            ):
                raise ValueError(f"invalid {size}x{size} shard group shape")
            cells = np.arange(area)
            spatial = (
                (packed_spatial[:, :, cells // 8] >> (cells % 8)) & 1
            ).reshape(rows, 3, size, size).astype(np.float32)
            ownership = (
                (packed_ownership[:, cells // 4] >> (2 * (cells % 4))) & 3
            ).reshape(rows, size, size)
            if np.any(ownership > 2):
                raise ValueError(f"invalid {size}x{size} ownership label")
            totals = policy.sum(axis=1, keepdims=True)
            if np.any(totals <= 0):
                raise ValueError(f"invalid {size}x{size} policy target")
            policy /= totals
            groups.append(
                BoardSamples(size, spatial, global_features, policy, value, ownership)
            )
    return groups
