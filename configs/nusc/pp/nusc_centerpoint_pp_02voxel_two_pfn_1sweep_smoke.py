from nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms import *  # noqa: F401,F403


nsweeps = 1
train_anno = "data/nuscenes/infos_train_01sweeps_withvelo_filter_True.pkl"
val_anno = "data/nuscenes/infos_val_01sweeps_withvelo_filter_True.pkl"

db_sampler["db_info_path"] = "data/nuscenes/dbinfos_train_1sweeps_withvelo.pkl"
db_sampler["enable"] = False
train_preprocessor["db_sampler"] = None

data["samples_per_gpu"] = 1
data["workers_per_gpu"] = 0

data["train"]["info_path"] = train_anno
data["train"]["ann_file"] = train_anno
data["train"]["nsweeps"] = nsweeps
data["train"]["load_interval"] = 20

data["val"]["info_path"] = val_anno
data["val"]["ann_file"] = val_anno
data["val"]["nsweeps"] = nsweeps
data["val"]["load_interval"] = 10

data["test"]["info_path"] = val_anno
data["test"]["ann_file"] = val_anno
data["test"]["nsweeps"] = nsweeps
data["test"]["load_interval"] = 10

total_epochs = 1
log_config["interval"] = 1
checkpoint_config["interval"] = 1
device_ids = range(1)
work_dir = "./work_dirs/nusc_pp_1sweep_smoke"
