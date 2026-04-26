from .. import builder
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        temporal_fusion=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.temporal_fusion = (
            builder.build_neck(temporal_fusion) if temporal_fusion is not None else None
        )

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def _extract_history_feat(self, example, history_num):
        history_data = dict(
            features=example["history_voxels"],
            num_voxels=example["history_num_points"],
            coors=example["history_coordinates"],
            batch_size=len(example["history_num_voxels"]),
            input_shape=example["shape"][0],
        )
        history = self.extract_feat(history_data)
        batch_size = len(example["num_voxels"])
        return history.reshape(batch_size, history_num, *history.shape[1:])

    def _apply_temporal_fusion(self, example, x, batch_size):
        if self.temporal_fusion is None:
            return x

        required_keys = [
            "history_voxels",
            "history_num_points",
            "history_num_voxels",
            "history_coordinates",
        ]
        missing_keys = [key for key in required_keys if key not in example]
        if missing_keys:
            raise KeyError(
                "temporal_fusion is enabled, but the batch is missing "
                f"history tensors: {missing_keys}. Set temporal_history_num "
                "in the dataset config and regenerate infos with enough sweeps."
            )

        total_history = len(example["history_num_voxels"])
        if total_history % batch_size != 0:
            raise ValueError(
                "history_num_voxels length must be divisible by batch size: "
                f"{total_history} vs {batch_size}"
            )
        history_num = int(total_history // batch_size)
        history = self._extract_history_feat(example, history_num)
        return self.temporal_fusion(
            x, history, history_transforms=example.get("history_frame_transforms")
        )

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        x = self._apply_temporal_fusion(example, x, batch_size)
        preds, _ = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        x = self._apply_temporal_fusion(example, x, batch_size)
        bev_feature = x
        preds, _ = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, None 
