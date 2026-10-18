"""Temporary network for exercising the V0 tensor boundary in tests."""

import torch
from torch import nn


class DummyModel(nn.Module):
    """Produces contract-shaped outputs; not the production architecture."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(5, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.board_policy = nn.Conv2d(8, 1, 1)
        self.pass_policy = nn.Linear(8, 1)
        self.value = nn.Linear(8, 3)
        self.ownership = nn.Conv2d(8, 1, 1)

    def forward(
        self, spatial: torch.Tensor, global_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_planes = global_features[:, :, None, None].expand(
            -1, -1, spatial.shape[2], spatial.shape[3]
        )
        features = self.trunk(torch.cat((spatial, global_planes), dim=1))
        pooled = features.mean(dim=(2, 3))
        policy_logits = torch.cat(
            (self.board_policy(features).flatten(1), self.pass_policy(pooled)), dim=1
        )
        return policy_logits, self.value(pooled), self.ownership(features)
