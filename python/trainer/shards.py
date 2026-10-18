"""Load the server's packed version-1 sampled NPZ shards."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PackedBoardSamples:
    board_size: int
    spatial: np.ndarray
    global_features: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    ownership: np.ndarray


def load_shard(path: Path) -> list[PackedBoardSamples]:
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
            policy = shard[prefix + "policy"]
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
            groups.append(
                PackedBoardSamples(
                    size, packed_spatial, global_features, policy, value, packed_ownership
                )
            )
    return groups
