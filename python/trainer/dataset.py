"""Index one packed shard as board-homogeneous training minibatches."""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .shards import PackedBoardSamples, load_shard


class BoardBatch(NamedTuple):
    board_size: int
    spatial: torch.Tensor
    global_features: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    ownership: torch.Tensor


class ShardDataset(Dataset):
    """Hold one packed shard; expand only the minibatch requested by the loader."""

    def __init__(self, path: Path, base_batch_size: int, effective_batch_size: int):
        if base_batch_size < 1:
            raise ValueError("base_batch_size must be positive")
        if effective_batch_size < 1 or effective_batch_size & (effective_batch_size - 1):
            raise ValueError("effective_batch_size must be a power of two")
        self.groups: list[PackedBoardSamples] = load_shard(path)
        self.batches: list[tuple[int, int, int]] = []
        rows_in_step = 0
        for group_index, group in enumerate(self.groups):
            max_microbatch_size = (
                base_batch_size * 19 * 19 // (group.board_size * group.board_size)
            )
            batch_size = 1 << (
                min(max_microbatch_size, effective_batch_size).bit_length() - 1
            )
            start = 0
            while start < len(group.policy):
                stop = min(
                    start + batch_size,
                    start + effective_batch_size - rows_in_step,
                    len(group.policy),
                )
                self.batches.append((group_index, start, stop))
                rows_in_step = (rows_in_step + stop - start) % effective_batch_size
                start = stop

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> BoardBatch:
        group_index, start, stop = self.batches[index]
        group = self.groups[group_index]
        size = group.board_size
        rows = stop - start
        area = size * size
        cells = np.arange(area)
        spatial = group.spatial[start:stop]
        ownership = group.ownership[start:stop]
        policy = group.policy[start:stop].astype(np.float32)
        spatial = ((spatial[:, :, cells // 8] >> (cells % 8)) & 1).reshape(
            rows, 3, size, size
        ).astype(np.float32)
        ownership = ((ownership[:, cells // 4] >> (2 * (cells % 4))) & 3).reshape(
            rows, size, size
        )
        if np.any(ownership > 2):
            raise ValueError(f"invalid {size}x{size} ownership label")
        totals = policy.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError(f"invalid {size}x{size} policy target")
        policy /= totals
        return BoardBatch(
            size,
            torch.from_numpy(spatial),
            torch.from_numpy(group.global_features[start:stop]),
            torch.from_numpy(policy),
            torch.from_numpy(group.value[start:stop]),
            torch.from_numpy(ownership.astype(np.int64)),
        )
