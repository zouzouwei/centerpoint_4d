# CenterPoint TemporalBEVFusion 训练说明

本文档记录当前仓库在 `centerpoint` conda 环境下新增的多帧 BEV 特征融合方案，以及如何和原始单帧训练做快速对比。

当前推荐方案是 **keyframe temporal BEV fusion**：不使用 nuScenes sweep 拼点增强当前帧，而是当前关键帧和前两个关键帧分别提 BEV feature，在 `CenterHead` 解码前做时序融合。

## 当前环境

当前已按 PyTorch 2 / CUDA 12 系列环境迁移，建议环境为：

```text
conda env: centerpoint
Python: 3.12
PyTorch: 2.6.0+cu124
CUDA extension: DCN 已可编译
Sparse convolution: 如需 voxel backbone，使用 spconv-cu124；PointPillars smoke 配置不依赖 spconv
```

基础环境安装和 CUDA 12 迁移细节见：

```text
README_PY312_CUDA124.md
requirements-py312.txt
```

## 实现方案

新增模块：

```text
det3d/models/necks/temporal_bev_fusion.py
```

推荐的 keyframe BEV warp 逻辑：

```text
current keyframe points -> voxelize -> PillarFeatureNet -> Scatter -> RPN -> current BEV
prev keyframe 1 points  -> voxelize -> PillarFeatureNet -> Scatter -> RPN -> history BEV 1
prev keyframe 2 points  -> voxelize -> PillarFeatureNet -> Scatter -> RPN -> history BEV 2
history BEV features    -> ego-motion BEV warp to current frame
current/history BEV     -> TemporalBEVFusion -> CenterHead
```

`TemporalBEVFusion` 默认使用 `current + T=2`：

```text
输入: current_bev [B, C, H, W]
输入: history_bev [B, 2, C, H, W]
融合: concat(current, history1, history2) -> 1x1 conv -> 3x3 conv
输出: current + sigmoid(gate) * fused
```

这个版本是 BEV 特征层融合，不改变原来的 detection head。没有配置 `model["temporal_fusion"]` 时，`PointPillars` 完全按原始单帧逻辑运行。

## 重要设计边界

nuScenes 常用 `sweeps` 是为了把多帧 LiDAR 点云拼到当前帧，缓解单帧点云稀疏问题。这个操作和 TemporalBEVFusion 不是一回事。

当前推荐配置不做 sweep 点云拼帧：

```text
nsweeps = 1
temporal_history_source = "keyframe"
temporal_history_align = "bev"
TemporalBEVFusion.align_history = True
```

也就是说，每一帧 LiDAR 仍然是单帧稀疏点云；前两帧只作为独立历史帧提特征，历史 BEV 在 feature map 层通过 `grid_sample` warp 到当前坐标系，然后在 `CenterHead` 前融合。

这更接近 BEVDet4D 的核心思想。和完整 BEVDet4D 的区别是：当前实现训练时直接读取前两帧并提特征，还没有做在线推理时的 feature cache。后续如果要优化推理速度，可以把上一帧 BEV 缓存起来，不重复跑 backbone。

## 数据索引

当前数据目录：

```text
data/nuscenes
```

已经生成：

```text
data/nuscenes/infos_train_01sweeps_withvelo_filter_True.pkl
data/nuscenes/infos_val_01sweeps_withvelo_filter_True.pkl
data/nuscenes/dbinfos_train_1sweeps_withvelo.pkl
data/nuscenes/gt_database_1sweeps_withvelo/

data/nuscenes/infos_train_03sweeps_withvelo_filter_True.pkl
data/nuscenes/infos_val_03sweeps_withvelo_filter_True.pkl
data/nuscenes/dbinfos_train_3sweeps_withvelo.pkl
data/nuscenes/gt_database_3sweeps_withvelo/
```

更新后的 `01sweeps` info 里已经额外写入：

```text
history_frames: 前两个关键帧的 lidar_path、time_lag、current_from_history transform_matrix
```

如果需要重新生成：

```bash
conda activate centerpoint
cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d
export PYTHONPATH="$PWD:${PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-centerpoint

python tools/create_data.py nuscenes_data_prep \
  --root_path=data/nuscenes \
  --version=v1.0-trainval \
  --nsweeps=1

python tools/create_data.py nuscenes_data_prep \
  --root_path=data/nuscenes \
  --version=v1.0-trainval \
  --nsweeps=3
```

推荐 keyframe temporal 配置只需要 `nsweeps=1`。`nsweeps=3` 只用于 sweep 兼容实验或点级拼帧对照实验。

## 对比配置

原始单帧 baseline：

```text
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_1sweep_smoke.py
```

推荐 TemporalBEVFusion：

```text
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_1sweep_keyframe_temporal_smoke.py
```

这个配置使用 `nsweeps=1`，历史帧来自前两个 keyframe，不使用 sweep 拼点。

兼容的 sweep temporal 配置：

```text
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_3sweep_temporal_smoke.py
```

这个配置会使用 nuScenes `sweeps` 作为历史输入，只适合做兼容或 ablation，不是当前推荐方案。

两套 smoke 配置都用于快速链路验证：

```text
samples_per_gpu = 1
workers_per_gpu = 0
load_interval: train=20, val=10
total_epochs = 1
GT database augmentation disabled
```

正式实验时建议复制这两个配置，调大训练 epoch、取消过大的 `load_interval`，并固定随机种子做多次对比。

## CPU smoke 验证

单帧 baseline forward 已验证通过：

```bash
conda run -n centerpoint env \
  PYTHONPATH=/home/hy/hycode/auto_driver/lidar/centerpoint_4d \
  MPLCONFIGDIR=/tmp/matplotlib-centerpoint \
  python -c "import torch; torch.set_num_threads(2); from det3d.torchie import Config; from det3d.datasets import build_dataset; from det3d.models import build_detector; from det3d.torchie.parallel import collate_kitti; from det3d.torchie.apis.train import example_to_device; cfg=Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_1sweep_smoke.py'); ds=build_dataset(cfg.data.train); ex=ds[0]; print('has_history', 'history_voxels' in ex); model=build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg); model.train(); batch=collate_kitti([ex]); example=example_to_device(batch, torch.device('cpu')); losses=model(example, return_loss=True); print({k:(len(v) if isinstance(v, list) else type(v).__name__) for k,v in losses.items()}); print('loss0', float(losses['loss'][0].detach().cpu()))"
```

当前结果：

```text
has_history False
{'loss': 6, 'hm_loss': 6, 'loc_loss': 6, 'loc_loss_elem': 6, 'num_positive': 6}
loss0 12.103281021118164
```

推荐 keyframe TemporalBEVFusion forward 已验证通过：

```bash
conda run -n centerpoint env \
  PYTHONPATH=/home/hy/hycode/auto_driver/lidar/centerpoint_4d \
  MPLCONFIGDIR=/tmp/matplotlib-centerpoint \
  python -c "import torch; torch.set_num_threads(2); from det3d.torchie import Config; from det3d.datasets import build_dataset; from det3d.models import build_detector; from det3d.torchie.parallel import collate_kitti; from det3d.torchie.apis.train import example_to_device; cfg=Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_1sweep_keyframe_temporal_smoke.py'); ds=build_dataset(cfg.data.train); model=build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg); model.train(); batch=collate_kitti([ds[0]]); example=example_to_device(batch, torch.device('cpu')); losses=model(example, return_loss=True); print({k:(len(v) if isinstance(v, list) else type(v).__name__) for k,v in losses.items()}); print('loss0', float(losses['loss'][0].detach().cpu()))"
```

当前结果：

```text
{'loss': 6, 'hm_loss': 6, 'loc_loss': 6, 'loc_loss_elem': 6, 'num_positive': 6}
loss0 17.651512145996094
```

## GPU smoke 训练

单帧 baseline：

```bash
conda activate centerpoint
cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d
export PYTHONPATH="$PWD:${PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-centerpoint

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_1sweep_smoke.py \
  --gpus 1
```

推荐 TemporalBEVFusion：

```bash
conda activate centerpoint
cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d
export PYTHONPATH="$PWD:${PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-centerpoint

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_1sweep_keyframe_temporal_smoke.py \
  --gpus 1
```

输出目录：

```text
work_dirs/nusc_pp_1sweep_smoke
work_dirs/nusc_pp_1sweep_keyframe_temporal_smoke
```

当前 Codex 沙箱无法可靠访问 GPU，所以这里完成的是 CPU 数据和 forward 验证。GPU 训练请在你的终端直接执行上面的命令。

## 正式对比建议

为了评估加入 T+2 历史帧是否提升检测率，建议先做三组：

```text
baseline: 当前帧 1-sweep，无 temporal_fusion
temporal: 当前 keyframe + 前 2 个 keyframe，BEV warp + TemporalBEVFusion
control: 3-sweep 点级拼接，无 temporal_fusion
```

其中前两组是当前已经提供的配置。第三组可以复用原始 10-sweep 配置改成 `nsweeps=3`，不设置 `temporal_fusion`，用于判断收益来自“更多点”还是“BEV 特征层时序融合”。
