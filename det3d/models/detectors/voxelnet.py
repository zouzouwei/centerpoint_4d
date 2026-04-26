from .. import builder
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
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
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.temporal_fusion = (
            builder.build_neck(temporal_fusion) if temporal_fusion is not None else None
        )
        
    def extract_feat(self, data):
        if 'voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def _extract_history_feat(self, example, history_num):
        history_batch_size = len(example["history_num_voxels"])
        history_data = dict(
            voxels=example["history_voxels"],
            num_points=example["history_num_points"],
            coordinates=example["history_coordinates"],
            points=[None] * history_batch_size,
            shape=example["shape"],
        )
        history, _ = self.extract_feat(history_data)
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
                "in the dataset config and provide keyframe/sweep history."
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
        x, _ = self.extract_feat(example)
        x = self._apply_temporal_fusion(example, x, len(example["num_voxels"]))
        preds, _ = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        x = self._apply_temporal_fusion(example, x, len(example["num_voxels"]))
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 
