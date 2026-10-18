"""Train on one packed shard with board-size-limited microbatches."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import ShardDataset


def train_shard(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    base_batch_size: int,
    effective_batch_size: int,
) -> int:
    """Train one shard and return its row count after all optimizer steps finish."""
    dataset = ShardDataset(path, base_batch_size, effective_batch_size)
    loader = DataLoader(dataset, batch_size=None)
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rows_in_step = 0
    trained_rows = 0
    for index, batch in enumerate(loader):
        spatial = batch.spatial.to(device)
        global_features = batch.global_features.to(device)
        policy_target = batch.policy.to(device)
        value_target = batch.value.to(device)
        ownership_target = batch.ownership.to(device)
        policy_logits, value, ownership_logits = model(spatial, global_features)

        policy_loss = -(
            policy_target * F.log_softmax(policy_logits, dim=1)
        ).sum(dim=1).mean()
        win_loss = F.binary_cross_entropy_with_logits(value[:, 0], value_target[:, 0])
        # The model predicts score mean and deviation in units of 20 points.
        score_loss = F.gaussian_nll_loss(
            value[:, 1], value_target[:, 1] / 20, F.softplus(value[:, 2]).square()
        )
        # Packed labels 0/1/2 mean opponent/neutral/player; inference uses tanh.
        ownership_loss = F.mse_loss(
            torch.tanh(ownership_logits[:, 0]), ownership_target.float() - 1
        )
        rows = len(policy_target)
        trained_rows += rows
        loss = policy_loss + win_loss + score_loss + ownership_loss
        # Sum row losses now; average once across the whole optimizer update.
        (loss * rows).backward()
        rows_in_step += rows

        if rows_in_step == effective_batch_size or index + 1 == len(loader):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(rows_in_step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            rows_in_step = 0
    return trained_rows
