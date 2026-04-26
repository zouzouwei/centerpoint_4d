from nusc_centerpoint_pp_02voxel_two_pfn_1sweep_smoke import *  # noqa: F401,F403


# Recommended temporal experiment: keep each LiDAR frame sparse, extract BEV
# features per keyframe, then fuse current + two previous keyframe BEV features
# immediately before CenterHead.
temporal_history_num = 2

model["temporal_fusion"] = dict(
    type="TemporalBEVFusion",
    in_channels=sum([128, 128, 128]),
    num_history=temporal_history_num,
    detach_history=False,
    use_gate=True,
    align_history=True,
    pc_range=voxel_generator["range"],
)

data["train"]["temporal_history_num"] = temporal_history_num
data["train"]["temporal_history_source"] = "keyframe"
data["train"]["temporal_history_align"] = "bev"

data["val"]["temporal_history_num"] = temporal_history_num
data["val"]["temporal_history_source"] = "keyframe"
data["val"]["temporal_history_align"] = "bev"

data["test"]["temporal_history_num"] = temporal_history_num
data["test"]["temporal_history_source"] = "keyframe"
data["test"]["temporal_history_align"] = "bev"

work_dir = "./work_dirs/nusc_pp_1sweep_keyframe_temporal_smoke"
