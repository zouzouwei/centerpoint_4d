from nusc_centerpoint_pp_02voxel_two_pfn_10sweep_circular_nms import *  # noqa: F401,F403


# Current frame + two historical LiDAR sweeps. The history sweeps are kept
# separate until BEV feature extraction, then fused by TemporalBEVFusion.
nsweeps = 3
temporal_history_num = 2
data_root = "data/nuscenes"
train_anno = "data/nuscenes/infos_train_03sweeps_withvelo_filter_True.pkl"
val_anno = "data/nuscenes/infos_val_03sweeps_withvelo_filter_True.pkl"

model["temporal_fusion"] = dict(
    type="TemporalBEVFusion",
    in_channels=sum([128, 128, 128]),
    num_history=temporal_history_num,
    detach_history=False,
    use_gate=True,
)

db_sampler["db_info_path"] = "data/nuscenes/dbinfos_train_3sweeps_withvelo.pkl"
db_sampler["enable"] = False
train_preprocessor["db_sampler"] = None

data["samples_per_gpu"] = 1
data["workers_per_gpu"] = 0

data["train"]["info_path"] = train_anno
data["train"]["ann_file"] = train_anno
data["train"]["root_path"] = data_root
data["train"]["nsweeps"] = nsweeps
data["train"]["temporal_history_num"] = temporal_history_num
data["train"]["load_interval"] = 20

data["val"]["info_path"] = val_anno
data["val"]["ann_file"] = val_anno
data["val"]["root_path"] = data_root
data["val"]["nsweeps"] = nsweeps
data["val"]["temporal_history_num"] = temporal_history_num
data["val"]["load_interval"] = 10

data["test"]["info_path"] = val_anno
data["test"]["ann_file"] = val_anno
data["test"]["root_path"] = data_root
data["test"]["nsweeps"] = nsweeps
data["test"]["temporal_history_num"] = temporal_history_num
data["test"]["load_interval"] = 10

total_epochs = 1
log_config["interval"] = 1
checkpoint_config["interval"] = 1
device_ids = range(1)
work_dir = "./work_dirs/nusc_pp_3sweep_temporal_smoke"
