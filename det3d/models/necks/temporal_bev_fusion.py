import torch
from torch import nn
import torch.nn.functional as F

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
        align_history=False,
        pc_range=None,
    ):
        super().__init__()
        assert num_history > 0
        assert mode in ["concat"]
        self.num_history = num_history
        self.detach_history = detach_history
        self.use_gate = use_gate
        self.align_history = align_history
        self.pc_range = pc_range

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

    def _warp_history_bev(self, history_bev, current_from_history):
        if self.pc_range is None:
            raise ValueError("pc_range must be configured when align_history=True")

        batch, num_history, channels, height, width = history_bev.shape
        grid_dtype = torch.float32
        transforms = current_from_history.to(
            device=history_bev.device, dtype=grid_dtype
        ).reshape(batch * num_history, 4, 4)
        history_from_current = torch.linalg.inv(transforms)

        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        xs = torch.linspace(
            x_min + (x_max - x_min) / (2 * width),
            x_max - (x_max - x_min) / (2 * width),
            width,
            device=history_bev.device,
            dtype=grid_dtype,
        )
        ys = torch.linspace(
            y_min + (y_max - y_min) / (2 * height),
            y_max - (y_max - y_min) / (2 * height),
            height,
            device=history_bev.device,
            dtype=grid_dtype,
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        zeros = torch.zeros_like(xx)
        ones = torch.ones_like(xx)
        current_grid = torch.stack([xx, yy, zeros, ones], dim=-1).reshape(-1, 4)

        history_grid = torch.matmul(
            history_from_current,
            current_grid.t().unsqueeze(0).expand(batch * num_history, -1, -1),
        ).transpose(1, 2)
        hist_x = history_grid[..., 0].reshape(batch * num_history, height, width)
        hist_y = history_grid[..., 1].reshape(batch * num_history, height, width)

        grid_x = 2.0 * (hist_x - x_min) / (x_max - x_min) - 1.0
        grid_y = 2.0 * (hist_y - y_min) / (y_max - y_min) - 1.0
        sample_grid = torch.stack([grid_x, grid_y], dim=-1)

        history_flat = history_bev.reshape(batch * num_history, channels, height, width)
        warped = F.grid_sample(
            history_flat.to(dtype=grid_dtype),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped.reshape(batch, num_history, channels, height, width).to(
            dtype=history_bev.dtype
        )

    def forward(self, current_bev, history_bev, history_transforms=None):
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

        if self.align_history:
            if history_transforms is None:
                raise ValueError("history_transforms is required when align_history=True")
            history_bev = self._warp_history_bev(history_bev, history_transforms)

        history_flat = history_bev.reshape(batch, num_history * channels, height, width)
        fused = self.fuse(torch.cat([current_bev, history_flat], dim=1))

        if self.gate is None:
            return fused

        gate = self.gate(torch.cat([current_bev, fused], dim=1))
        return current_bev + gate * fused
