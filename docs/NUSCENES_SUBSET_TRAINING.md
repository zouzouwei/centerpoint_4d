# nuScenes trainval03 子集解压与 smoke 训练

本文档记录当前 `data/nuscenes` 下这三个包的处理方式：

```text
nuScenes-map-expansion-v1.3.zip
v1.0-trainval03_blobs.tgz
v1.0-trainval_meta.tgz
```

## 解压

在仓库根目录执行：

```bash
cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d

tar -xzf data/nuscenes/v1.0-trainval_meta.tgz -C data/nuscenes
python -c "import zipfile, pathlib; root=pathlib.Path('data/nuscenes'); zipfile.ZipFile(root/'nuScenes-map-expansion-v1.3.zip').extractall(root)"
tar -xzf data/nuscenes/v1.0-trainval03_blobs.tgz -C data/nuscenes
```

解压后的关键结构应为：

```text
data/nuscenes/
  samples/
  sweeps/
  maps/
  v1.0-trainval/
```

当前已解压并确认：

```text
samples 文件数: 40524
sweeps 文件数: 219394
```

## 生成 1-sweep 训练索引

`trainval03_blobs` 只是 trainval 的一个子包，不是完整 trainval。已修复 `det3d/datasets/nuscenes/nusc_common.py`，让 info 生成时跳过未下载的 scene。

执行：

```bash
conda activate centerpoint
cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d
export PYTHONPATH="$PWD:${PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-centerpoint

python tools/create_data.py nuscenes_data_prep \
  --root_path=data/nuscenes \
  --version=v1.0-trainval \
  --nsweeps=1
```

当前生成结果：

```text
exist scene num: 85
train scene: 73
val scene: 12
train sample: 2902
val sample: 475
data/nuscenes/infos_train_01sweeps_withvelo_filter_True.pkl
data/nuscenes/infos_val_01sweeps_withvelo_filter_True.pkl
data/nuscenes/dbinfos_train_1sweeps_withvelo.pkl
data/nuscenes/gt_database_1sweeps_withvelo/
```

## smoke 训练配置

新增配置：

```text
configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_1sweep_smoke.py
```

这个配置用于快速验证训练链路：

```text
nsweeps = 1
samples_per_gpu = 1
workers_per_gpu = 0
train load_interval = 20
val load_interval = 10
total_epochs = 1
GT database augmentation disabled
```

## 验证数据和 loss 前向

已通过：

```bash
conda run -n centerpoint env \
  PYTHONPATH=/home/hy/hycode/auto_driver/lidar/centerpoint_4d \
  MPLCONFIGDIR=/tmp/matplotlib-centerpoint \
  python -c "import torch; from det3d.torchie import Config; from det3d.datasets import build_dataset; from det3d.models import build_detector; from det3d.torchie.parallel import collate_kitti; from det3d.torchie.apis.train import example_to_device; cfg=Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_1sweep_smoke.py'); ds=build_dataset(cfg.data.train); model=build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg); model.train(); batch=collate_kitti([ds[0]]); example=example_to_device(batch, torch.device('cpu')); losses=model(example, return_loss=True); print({k:(len(v) if isinstance(v, list) else type(v).__name__) for k,v in losses.items()}); print('loss0', float(losses['loss'][0].detach().cpu()))"
```

当前输出包含：

```text
{'loss': 6, 'hm_loss': 6, 'loc_loss': 6, 'loc_loss_elem': 6, 'num_positive': 6}
loss0 13.681405067443848
```

## 启动 GPU smoke 训练

当前沙箱里 GPU 不可用，所以没有在这里跑完整 `tools/train.py`。你在能访问 GPU 的终端里执行：

```bash
conda activate centerpoint
cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d
export PYTHONPATH="$PWD:${PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-centerpoint

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_1sweep_smoke.py \
  --gpus 1
```

输出目录：

```text
work_dirs/nusc_pp_1sweep_smoke
```

这个配置只是验证链路，不用于正式指标。正式训练需要完整 trainval blobs、更多 sweep、更多 epoch 和 GPU。
