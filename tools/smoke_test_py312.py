import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-centerpoint")

from det3d.models import build_detector  # noqa: E402
from det3d.torchie import Config  # noqa: E402


def main():
    cfg = Config.fromfile(
        str(ROOT / "configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms.py")
    )
    cfg.test_cfg.score_threshold = 0.99

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    model.eval()

    pc_range = np.array(cfg.voxel_generator.range, dtype=np.float32)
    voxel_size = np.array(cfg.voxel_generator.voxel_size, dtype=np.float32)
    grid_size = np.round((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int64)

    max_points = cfg.voxel_generator.max_points_in_voxel
    num_features = cfg.model.reader.num_input_features
    example = {
        "voxels": torch.zeros((1, max_points, num_features), dtype=torch.float32),
        "coordinates": torch.zeros((1, 4), dtype=torch.int32),
        "num_points": torch.ones((1,), dtype=torch.int32),
        "num_voxels": torch.tensor([1], dtype=torch.int32),
        "shape": [grid_size.tolist()],
        "metadata": [],
    }

    with torch.no_grad():
        output = model(example, return_loss=False)

    pred = output[0]
    print(
        "smoke ok:",
        f"boxes={tuple(pred['box3d_lidar'].shape)}",
        f"scores={tuple(pred['scores'].shape)}",
        f"labels={tuple(pred['label_preds'].shape)}",
    )


if __name__ == "__main__":
    main()
