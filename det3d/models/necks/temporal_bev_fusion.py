import torch
from torch import nn

from ..registry import NECKS


@NECKS.register_module
class TemporalBEVFusion(nn.Module):
    """Fuse current BEV features with a fixed number of historical BEV features.

    The first implementation assumes historical point clouds have already been
    transformed to the current ego frame before voxelization.
    """

    def __init__(
        self,
        in_channels,
        num_history=2,
        mode="concat",
        detach_history=False,
        use_gate=True,
    ):
        super().__init__()
        assert num_history > 0
        assert mode in ["concat"]
        self.num_history = num_history
        self.detach_history = detach_history
        self.use_gate = use_gate

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * (num_history + 1), in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

    def forward(self, current_bev, history_bev):
        if history_bev is None:
            return current_bev

        if history_bev.dim() != 5:
            raise ValueError(
                "history_bev must have shape [B, T, C, H, W], "
                f"but got {tuple(history_bev.shape)}"
            )

        batch, num_history, channels, height, width = history_bev.shape
        if num_history != self.num_history:
            raise ValueError(f"expected {self.num_history} history frames, got {num_history}")
        if current_bev.shape != (batch, channels, height, width):
            raise ValueError(
                "current_bev and history_bev spatial/channel shapes do not match: "
                f"{tuple(current_bev.shape)} vs {tuple(history_bev.shape)}"
            )

        if self.detach_history:
            history_bev = history_bev.detach()

        history_flat = history_bev.reshape(batch, num_history * channels, height, width)
        fused = self.fuse(torch.cat([current_bev, history_flat], dim=1))

        if self.gate is None:
            return fused

        gate = self.gate(torch.cat([current_bev, fused], dim=1))
        return current_bev + gate * fused
