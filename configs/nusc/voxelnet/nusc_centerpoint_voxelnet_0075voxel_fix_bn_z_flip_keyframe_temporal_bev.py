from nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_flip import *  # noqa: F401,F403


# VoxelNet/SECOND version of keyframe BEV temporal fusion.
# Keep LiDAR input as single-frame (`nsweeps=1`), extract BEV features for the
# current keyframe and the two previous keyframes separately, warp historical
# BEV features by ego motion, then fuse immediately before CenterHead.
temporal_history_num = 2
nsweeps = 10
data_root = "data/nuscenes"
train_anno = "data/nuscenes/infos_train_10sweeps_withvelo_filter_True_30scenes.pkl"
val_anno = "data/nuscenes/infos_val_10sweeps_withvelo_filter_True_5scenes.pkl"

model["temporal_fusion"] = dict(
    type="TemporalBEVFusion",
    in_channels=sum([256, 256]),
    num_history=temporal_history_num,
    detach_history=False,
    use_gate=True,
    align_history=True,
    pc_range=voxel_generator["range"],
)

# The imported flip config creates double-flip test-time branches that do not
# carry history tensors. Disable it for temporal BEV fusion.
DOUBLE_FLIP = False
voxel_generator["double_flip"] = False
test_cfg["double_flip"] = False
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat", double_flip=False),
]

# GT database sampling stays disabled for this temporal config so the temporal
# history branches do not receive object samples that only exist in the current frame.
db_sampler["db_info_path"] = "data/nuscenes/dbinfos_train_10sweeps_withvelo.pkl"
db_sampler["enable"] = False
train_preprocessor["db_sampler"] = None

# Temporal VoxelNet runs three sparse backbones per sample, so start with batch
# size 1 and increase only after checking GPU memory.
data["samples_per_gpu"] = 1
data["workers_per_gpu"] = 0

data["train"]["root_path"] = data_root
data["train"]["info_path"] = train_anno
data["train"]["ann_file"] = train_anno
data["train"]["nsweeps"] = nsweeps
data["train"]["temporal_history_num"] = temporal_history_num
data["train"]["temporal_history_source"] = "keyframe"
data["train"]["temporal_history_align"] = "bev"

data["val"]["root_path"] = data_root
data["val"]["info_path"] = val_anno
data["val"]["ann_file"] = val_anno
data["val"]["nsweeps"] = nsweeps
data["val"]["temporal_history_num"] = temporal_history_num
data["val"]["temporal_history_source"] = "keyframe"
data["val"]["temporal_history_align"] = "bev"
data["val"]["pipeline"] = test_pipeline

data["test"]["root_path"] = data_root
data["test"]["info_path"] = val_anno
data["test"]["ann_file"] = val_anno
data["test"]["nsweeps"] = nsweeps
data["test"]["temporal_history_num"] = temporal_history_num
data["test"]["temporal_history_source"] = "keyframe"
data["test"]["temporal_history_align"] = "bev"
data["test"]["pipeline"] = test_pipeline
data["test"].pop("version", None)

device_ids = range(1)
work_dir = "./work_dirs/nusc_voxelnet_0075_keyframe_temporal_bev_30train_5val"
