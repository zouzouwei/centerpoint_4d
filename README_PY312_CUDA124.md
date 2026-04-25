# CenterPoint Python 3.12 / CUDA 12.4 迁移运行说明

本文档记录本仓库在当前 `centerpoint` conda 环境中的初步迁移结果。原始项目面向 Python 3.6、CUDA 10、旧版 PyTorch 和旧版 spconv；当前已验证的基础运行路径是 Python 3.12 + PyTorch 2.6 + PointPillars + circular NMS。

## 已验证环境

当前环境：

```bash
conda activate centerpoint
python -V
python -c "import torch, torchvision, numpy, numba, cv2, shapely; print(torch.__version__, torchvision.__version__, numpy.__version__, numba.__version__, cv2.__version__, shapely.__version__)"
```

本机已验证版本：

```text
Python 3.12.13
torch 2.6.0+cu124
torchvision 0.21.0+cu124
numpy 1.26.4
numba 0.65.1
opencv 4.10.0
shapely 2.0.7
```

当前沙箱内 `nvidia-smi` 无法访问 NVML，`torch.cuda.is_available()` 返回 `False`，并且未找到 `nvcc`。因此本次验证不包含真实 GPU 推理、训练，也没有本地编译 CUDA 扩展。

## 安装步骤

不要在 Python 3.12 环境中直接使用旧的 `requirements.txt`。旧文件里包含 `Pillow<=6.2.1`、`open3d-python`、`nuscenes-devkit==1.0.5` 等旧约束，会把依赖求解拖回不支持 Python 3.12 的包。

在当前已有的 `centerpoint` 环境中安装依赖：

```bash
conda activate centerpoint
cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d

python -m pip install --only-binary=:all: -r requirements-py312.txt
python -m pip install --no-deps nuscenes-devkit==1.2.0
```

如果你从零创建环境，先安装与当前环境一致的 PyTorch，再执行上面的依赖安装：

```bash
conda create -n centerpoint python=3.12 -y
conda activate centerpoint

python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d
python -m pip install --only-binary=:all: -r requirements-py312.txt
python -m pip install --no-deps nuscenes-devkit==1.2.0
```

`nuscenes-devkit` 需要单独 `--no-deps` 安装，是因为它的依赖求解在 Python 3.12 下可能回退到旧版 `matplotlib/shapely`，导致构建失败。兼容它的 `numpy==1.26.4`、`shapely==2.0.7`、`opencv-python-headless==4.10.0.84`、`parameterized==0.9.0` 已经在 `requirements-py312.txt` 中安装。

## 运行前环境变量

从仓库根目录运行脚本时建议设置：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-centerpoint
```

`PYTHONPATH` 用于让 `tools/*.py` 能导入本仓库的 `det3d` 包。`MPLCONFIGDIR` 用于避免只读 home 目录下 matplotlib 缓存目录创建失败。

## 初步运行验证

本次迁移新增了一个不依赖数据、checkpoint、GPU、spconv、自定义 CUDA 扩展的 smoke test：

```bash
conda activate centerpoint
cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d
python tools/smoke_test_py312.py
```

预期输出类似：

```text
No spconv, sparse convolution disabled!
Import spconv fail, no support for sparse convolution!
Deformable Convolution not built!
Use HM Bias:  -2.19
smoke ok: boxes=(0, 9) scores=(0,) labels=(0,)
```

这表示 PointPillars 模型已经可以在 CPU 上完成构建和一次前向推理。输出为空是正常的，因为 smoke test 使用的是全零伪输入，并把阈值提高到了 `0.99`。

也可以验证脚本入口：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-centerpoint

python -m pip check
python tools/train.py --help
python tools/simple_inference_waymo.py --help
```

## 当前可用和不可用范围

已验证可用：

- Python 3.12 环境基础依赖导入。
- `det3d.torchie.Config` 配置加载。
- `det3d.datasets` 数据集模块导入。
- `det3d.models.build_detector` 构建 PointPillars。
- `configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms.py` 的 CPU smoke forward。
- `tools/train.py --help` 和 `tools/simple_inference_waymo.py --help` 入口启动。
- 无 spconv 时，`tools/simple_inference_waymo.py` 会回退到仓库内置 CPU voxelizer。

未在当前机器验证：

- GPU 训练和 GPU 推理，因为当前沙箱无法访问 GPU。
- `det3d/ops/iou3d_nms` CUDA 扩展，因为当前环境没有 `nvcc`。
- `det3d/ops/dcn` CUDA 扩展。
- VoxelNet / SparseConv 配置，因为当前没有可用 spconv。
- Waymo 官方评测扩展和 TensorFlow/Waymo Open Dataset 依赖。

## CUDA 扩展说明

如果只跑 circular NMS 的 PointPillars 配置，可以不编译 `iou3d_nms_cuda`。如果使用非 circular NMS 配置，代码会在运行时提示：

```text
iou3d_nms_cuda is not built. Build det3d/ops/iou3d_nms or use a circular_nms config ...
```

有 `nvcc` 且 CUDA/PyTorch 版本匹配时，可以尝试：

```bash
conda activate centerpoint
cd /home/hy/hycode/auto_driver/lidar/centerpoint_4d

cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace

cd ../dcn
python setup.py build_ext --inplace
```

本仓库自带的 `spconv/` 是旧版源码，不建议在 Python 3.12 + PyTorch 2.6 下直接编译。若需要 VoxelNet/SparseConv 配置，优先安装与 CUDA 12 / PyTorch 2 匹配的 spconv 2.x wheel，然后验证：

```bash
python -c "import spconv.pytorch as spconv; print(spconv.SparseConv3d)"
```

spconv 验证通过后，`det3d.models` 会自动启用 sparse convolution backbone。

## 本次兼容性修改摘要

- 修复 `det3d.models` 和 `det3d.models.backbones` 对 spconv 的误检测，只有真正能导入 `SparseConv3d` 时才启用 sparse backbone。
- 让 `iou3d_nms_cuda` 变为可选导入；未编译时不再阻塞普通 PointPillars 模块导入。
- 在需要 rotated CUDA NMS 的函数里给出明确运行时错误，而不是导入阶段崩溃。
- `tools/simple_inference_waymo.py` 在没有旧版 `spconv.utils.VoxelGenerator` 时回退到仓库内置 voxelizer。
- 修复 `PillarFeatureNet` 单 voxel 输入下裸 `squeeze()` 导致维度被错误压缩的问题。
- 更新 PyTorch 2.x 下 `torch.meshgrid` 的 `indexing="ij"` 参数。
- 新增 `requirements-py312.txt` 和 `tools/smoke_test_py312.py`，用于复现当前基础运行环境。
