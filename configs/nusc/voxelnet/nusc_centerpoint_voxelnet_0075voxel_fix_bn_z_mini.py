from pathlib import Path

_base_cfg = Path(__file__).with_name("nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py")
exec(compile(_base_cfg.read_text(), str(_base_cfg), "exec"))

mini_version = "v1.0-mini"

data["samples_per_gpu"] = 1
data["workers_per_gpu"] = 2

data["train"]["version"] = mini_version
data["val"]["version"] = mini_version
data["test"]["version"] = mini_version

# Keep the generated mini info files under the same names as the upstream script.
data["train"]["info_path"] = train_anno
data["train"]["ann_file"] = train_anno
data["val"]["info_path"] = val_anno
data["val"]["ann_file"] = val_anno

total_epochs = 5
device_ids = range(1)
work_dir = "./work_dirs/{}/".format(__file__[__file__.rfind("/") + 1 : -3])
