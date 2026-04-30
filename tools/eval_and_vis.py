import argparse
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path

try:
    import apex
except Exception:
    apex = None
    print("No APEX!")

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt_of_sample_tokens,
    load_prediction_of_sample_tokens,
)
import torch
from torch.nn.parallel import DistributedDataParallel
from pyquaternion import Quaternion

sys.path.append(".")

from demo_utils import Box, view_points
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.datasets.nuscenes.nusc_common import CAM_CHANS, cls_attr_dist, _lidar_nusc_box_to_global
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import batch_processor, get_root_logger
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize


GT_COLOR_BGR = (0, 255, 0)
PRED_COLOR_BGR = (0, 0, 255)
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nuScenes validation and save BEV plus six-camera visualizations"
    )
    parser.add_argument("config", help="config file path")
    parser.add_argument("--work_dir", required=True, help="directory for eval outputs")
    parser.add_argument("--checkpoint", help="checkpoint to evaluate")
    parser.add_argument("--prediction_pkl", help="existing prediction.pkl to evaluate/visualize")
    parser.add_argument("--skip_eval", action="store_true", help="skip nuScenes metric evaluation")
    parser.add_argument("--vis_dir", help="directory for visualization images")
    parser.add_argument("--vis_num", type=int, default=100, help="number of samples to visualize; -1 for all")
    parser.add_argument("--score_thr", type=float, default=0.2, help="prediction score threshold for visualization")
    parser.add_argument("--bev_range", type=float, default=54.0, help="BEV axis range in meters")
    parser.add_argument("--camera_width", type=int, default=480, help="width for each camera tile")
    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs for non-distributed eval")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--speed_test", action="store_true")
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    if args.prediction_pkl is None and args.checkpoint is None:
        parser.error("--checkpoint is required unless --prediction_pkl is provided")
    return args


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def build_data(cfg, args, distributed):
    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = None
    if args.prediction_pkl is None:
        data_loader = build_dataloader(
            dataset,
            batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
    return dataset, data_loader


def run_inference(cfg, args, data_loader, distributed, logger):
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    if distributed:
        if apex is None:
            raise RuntimeError("Distributed eval requires apex in this repo's current test path")
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()

    model.eval()
    detections = {}
    cpu_device = torch.device("cpu")

    if cfg.local_rank == 0:
        total = max(len(data_loader.dataset) // cfg.gpus, 1)
        prog_bar = torchie.ProgressBar(total)

    start_time = time.time()
    for data_batch in data_loader:
        with torch.no_grad():
            outputs = batch_processor(
                model,
                data_batch,
                train_mode=False,
                local_rank=args.local_rank,
            )
        for output in outputs:
            token = output["metadata"]["token"]
            for key, value in list(output.items()):
                if key != "metadata" and hasattr(value, "to"):
                    output[key] = value.to(cpu_device)
            detections[token] = output
            if cfg.local_rank == 0:
                prog_bar.update()

    synchronize()
    all_predictions = all_gather(detections)

    if args.local_rank != 0:
        return None

    predictions = {}
    for pred in all_predictions:
        predictions.update(pred)
    logger.info("Inference finished in {:.2f}s".format(time.time() - start_time))
    return predictions



def detection_to_nusc_boxes(detection):
    if detection is None:
        return []
    box3d = tensor_to_numpy(detection["box3d_lidar"]).copy()
    scores = tensor_to_numpy(detection["scores"])
    labels = tensor_to_numpy(detection["label_preds"])
    if box3d.size == 0:
        return []

    boxes = []
    box3d[:, -1] = -box3d[:, -1] - np.pi / 2
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=float(box3d[i, -1]))
        velocity = (float(box3d[i, 6]), float(box3d[i, 7]), 0.0) if box3d.shape[1] >= 8 else (0.0, 0.0, 0.0)
        boxes.append(
            Box(
                list(box3d[i, :3]),
                list(box3d[i, 3:6]),
                quat,
                label=int(labels[i]),
                score=float(scores[i]),
                velocity=velocity,
            )
        )
    return boxes


def format_nusc_results(dataset, predictions, output_dir):
    nusc = NuScenes(version=dataset.version, dataroot=str(dataset._root_path), verbose=True)
    mapped_class_names = []
    for name in dataset._class_names:
        mapped_class_names.append(dataset._name_mapping.get(name, name))

    nusc_annos = {"results": {}, "meta": None}
    for info in dataset._nusc_infos:
        token = info["token"]
        det = predictions[token]
        annos = []
        boxes = detection_to_nusc_boxes(det)
        boxes = _lidar_nusc_box_to_global(nusc, boxes, token)
        for box in boxes:
            name = mapped_class_names[box.label]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ["car", "construction_vehicle", "bus", "truck", "trailer"]:
                    attr = "vehicle.moving"
                elif name in ["bicycle", "motorcycle"]:
                    attr = "cycle.with_rider"
                else:
                    attr = None
            else:
                if name in ["pedestrian"]:
                    attr = "pedestrian.standing"
                elif name in ["bus"]:
                    attr = "vehicle.stopped"
                else:
                    attr = None

            annos.append(
                {
                    "sample_token": token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr if attr is not None else max(cls_attr_dist[name].items(), key=lambda item: item[1])[0],
                }
            )
        nusc_annos["results"][token] = annos

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    info_name = Path(dataset._info_path).stem
    result_path = Path(output_dir) / (info_name + ".json")
    with open(result_path, "w") as f:
        json.dump(nusc_annos, f)
    print("Finish generate predictions, save to {}".format(result_path))
    return nusc, result_path, mapped_class_names


def summarize_nusc_metrics(metrics, mapped_class_names):
    detail = {}
    result = "Nusc {} Evaluation\n".format("subset")
    for name in mapped_class_names:
        detail[name] = {}
        for key, value in metrics["label_aps"][name].items():
            detail[name]["dist@{}".format(key)] = value
        threshs = ", ".join([str(key) for key in metrics["label_aps"][name].keys()])
        scores = list(metrics["label_aps"][name].values())
        mean = sum(scores) / len(scores)
        score_text = ", ".join(["{:.2f}".format(score * 100) for score in scores])
        result += "{} Nusc dist AP@{}\n".format(name, threshs)
        result += score_text
        result += " mean AP: {}\n".format(mean)
    return {"results": {"nusc": result}, "detail": {"eval.nusc": detail}, "metrics_summary": metrics}


def evaluate_subset(dataset, predictions, output_dir, testset=False):
    if testset:
        return dataset.evaluation(copy.deepcopy(predictions), output_dir=output_dir, testset=True)[0]

    missing = [info["token"] for info in dataset._nusc_infos if info["token"] not in predictions]
    if missing:
        raise KeyError("Missing predictions for {} dataset samples, first token: {}".format(len(missing), missing[0]))

    nusc, result_path, mapped_class_names = format_nusc_results(dataset, predictions, output_dir)
    cfg = config_factory(dataset.eval_version)
    sample_tokens = [info["token"] for info in dataset._nusc_infos]

    nusc_eval = object.__new__(NuScenesEval)
    nusc_eval.nusc = nusc
    nusc_eval.result_path = str(result_path)
    nusc_eval.eval_set = "current_val_subset"
    nusc_eval.output_dir = output_dir
    nusc_eval.verbose = True
    nusc_eval.cfg = cfg
    nusc_eval.plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(nusc_eval.plot_dir, exist_ok=True)

    print("Initializing nuScenes detection evaluation for {} current dataset samples".format(len(sample_tokens)))
    nusc_eval.pred_boxes, nusc_eval.meta = load_prediction_of_sample_tokens(
        str(result_path),
        cfg.max_boxes_per_sample,
        DetectionBox,
        sample_tokens=sample_tokens,
        verbose=True,
    )
    nusc_eval.gt_boxes = load_gt_of_sample_tokens(nusc, sample_tokens, DetectionBox, verbose=True)
    assert set(nusc_eval.pred_boxes.sample_tokens) == set(nusc_eval.gt_boxes.sample_tokens)

    nusc_eval.pred_boxes = add_center_dist(nusc, nusc_eval.pred_boxes)
    nusc_eval.gt_boxes = add_center_dist(nusc, nusc_eval.gt_boxes)
    print("Filtering predictions")
    nusc_eval.pred_boxes = filter_eval_boxes(nusc, nusc_eval.pred_boxes, cfg.class_range, verbose=True)
    print("Filtering ground truth annotations")
    nusc_eval.gt_boxes = filter_eval_boxes(nusc, nusc_eval.gt_boxes, cfg.class_range, verbose=True)
    nusc_eval.sample_tokens = nusc_eval.gt_boxes.sample_tokens

    metrics = nusc_eval.main(plot_examples=0, render_curves=True)
    return summarize_nusc_metrics(metrics, mapped_class_names)


def prediction_to_boxes(detection, score_thr):
    if detection is None:
        return []
    box3d = tensor_to_numpy(detection["box3d_lidar"]).copy()
    scores = tensor_to_numpy(detection["scores"])
    labels = tensor_to_numpy(detection["label_preds"])
    if box3d.size == 0:
        return []

    boxes = []
    yaws = -box3d[:, -1] - np.pi / 2
    for i in range(box3d.shape[0]):
        score = float(scores[i])
        if score < score_thr:
            continue
        quat = Quaternion(axis=[0, 0, 1], radians=float(yaws[i]))
        velocity = (float(box3d[i, 6]), float(box3d[i, 7]), 0.0) if box3d.shape[1] >= 8 else (0.0, 0.0, 0.0)
        boxes.append(
            Box(
                list(box3d[i, :3]),
                list(box3d[i, 3:6]),
                quat,
                label=int(labels[i]),
                score=score,
                velocity=velocity,
            )
        )
    return boxes


def gt_to_boxes(info):
    if "gt_boxes" not in info:
        return []
    gt_boxes = np.asarray(info["gt_boxes"])
    gt_names = np.asarray(info.get("gt_names", [None] * len(gt_boxes)))
    boxes = []
    for i, gt_box in enumerate(gt_boxes):
        name = gt_names[i] if i < len(gt_names) else None
        if name == "ignore":
            continue
        yaw = -float(gt_box[-1]) - np.pi / 2
        quat = Quaternion(axis=[0, 0, 1], radians=yaw)
        velocity = (float(gt_box[6]), float(gt_box[7]), 0.0) if gt_box.shape[0] >= 8 else (0.0, 0.0, 0.0)
        boxes.append(
            Box(
                list(gt_box[:3]),
                list(gt_box[3:6]),
                quat,
                label=np.nan,
                score=np.nan,
                velocity=velocity,
                name=str(name),
            )
        )
    return boxes


def resolve_path(root_path, path):
    path = Path(path)
    if path.exists():
        return path
    candidate = Path(root_path) / path
    if candidate.exists():
        return candidate
    return path


def read_lidar_points(info, root_path):
    lidar_path = resolve_path(root_path, info["lidar_path"])
    if not lidar_path.exists():
        return None
    raw = np.fromfile(str(lidar_path), dtype=np.float32)
    if raw.size % 5 != 0:
        return None
    return raw.reshape(-1, 5)[:, :3]


def render_bev(points, gt_boxes, pred_boxes, bev_range):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=160)

    if points is not None and len(points) > 0:
        mask = (np.abs(points[:, 0]) <= bev_range) & (np.abs(points[:, 1]) <= bev_range)
        points = points[mask]
        if len(points) > 0:
            dists = np.sqrt(np.sum(points[:, :2] ** 2, axis=1))
            colors = np.minimum(1.0, dists / bev_range)
            ax.scatter(points[:, 0], points[:, 1], c=colors, s=0.15, cmap="gray")

    for box in gt_boxes:
        box.render(ax, view=np.eye(4), colors=("lime", "lime", "lime"), linewidth=1.4)
    for box in pred_boxes:
        box.render(ax, view=np.eye(4), colors=("red", "red", "red"), linewidth=1.0)

    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x / forward (m)")
    ax.set_ylabel("y / left (m)")
    ax.set_title("BEV  GT=green  Pred=red")
    ax.grid(True, linewidth=0.2, alpha=0.35)
    fig.tight_layout(pad=0.2)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    image = image[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def draw_projected_box(image, box, lidar_to_cam, intrinsic, color, linewidth):
    corners = box.corners()
    corners_hom = np.vstack((corners, np.ones((1, corners.shape[1]))))
    corners_cam = np.asarray(lidar_to_cam).dot(corners_hom)[:3, :]
    depths = corners_cam[2, :]
    if np.any(depths <= 0.1):
        return

    corners_img = view_points(corners_cam, np.asarray(intrinsic), normalize=True)[:2, :].T
    if not np.isfinite(corners_img).all():
        return

    height, width = image.shape[:2]
    if (
        corners_img[:, 0].max() < 0
        or corners_img[:, 0].min() >= width
        or corners_img[:, 1].max() < 0
        or corners_img[:, 1].min() >= height
    ):
        return

    limit = max(width, height) * 5
    if np.abs(corners_img).max() > limit:
        return

    corners_img = corners_img.astype(np.int32)
    for start, end in BOX_EDGES:
        cv2.line(
            image,
            tuple(corners_img[start]),
            tuple(corners_img[end]),
            color,
            linewidth,
            lineType=cv2.LINE_AA,
        )


def resize_to_width(image, width):
    height = int(round(image.shape[0] * float(width) / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def render_camera_tile(path, cam_name, lidar_to_cam, intrinsic, gt_boxes, pred_boxes, camera_width):
    image = cv2.imread(str(path))
    if image is None:
        image = np.zeros((270, camera_width, 3), dtype=np.uint8)
        cv2.putText(image, "missing image", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        for box in gt_boxes:
            draw_projected_box(image, box, lidar_to_cam, intrinsic, GT_COLOR_BGR, 2)
        for box in pred_boxes:
            draw_projected_box(image, box, lidar_to_cam, intrinsic, PRED_COLOR_BGR, 2)
        image = resize_to_width(image, camera_width)

    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(image, cam_name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def pad_to_shape(image, height, width):
    if image.shape[0] == height and image.shape[1] == width:
        return image
    out = np.zeros((height, width, 3), dtype=image.dtype)
    out[: image.shape[0], : image.shape[1]] = image
    return out


def compose_camera_panel(tiles):
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    tiles = [pad_to_shape(tile, tile_h, tile_w) for tile in tiles]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    return np.vstack(rows)


def visualize_sample(info, prediction, dataset, output_path, args):
    root_path = getattr(dataset, "_root_path", ".")
    gt_boxes = gt_to_boxes(info)
    pred_boxes = prediction_to_boxes(prediction, args.score_thr)
    points = read_lidar_points(info, root_path)

    bev = render_bev(points, gt_boxes, pred_boxes, args.bev_range)

    cam_paths = info.get("all_cams_path", [])
    cam_intrinsics = info.get("all_cams_intrinsic", [])
    cam_transforms = info.get("all_cams_from_lidar", [])
    tiles = []
    for cam_name, cam_path, intrinsic, transform in zip(CAM_CHANS, cam_paths, cam_intrinsics, cam_transforms):
        path = resolve_path(root_path, cam_path)
        tiles.append(
            render_camera_tile(
                path,
                cam_name,
                transform,
                intrinsic,
                gt_boxes,
                pred_boxes,
                args.camera_width,
            )
        )

    if len(tiles) != 6:
        blank = np.zeros((270, args.camera_width, 3), dtype=np.uint8)
        while len(tiles) < 6:
            tiles.append(blank.copy())
    camera_panel = compose_camera_panel(tiles[:6])

    bev = cv2.resize(bev, (camera_panel.shape[0], camera_panel.shape[0]), interpolation=cv2.INTER_AREA)
    canvas = np.hstack([bev, camera_panel])
    cv2.imwrite(str(output_path), canvas)


def visualize_predictions(dataset, predictions, args):
    vis_dir = Path(args.vis_dir or Path(args.work_dir) / "vis")
    vis_dir.mkdir(parents=True, exist_ok=True)

    infos = dataset._nusc_infos
    vis_count = len(infos) if args.vis_num < 0 else min(args.vis_num, len(infos))
    saved = 0
    for idx in range(vis_count):
        info = infos[idx]
        token = info["token"]
        prediction = predictions.get(token)
        if prediction is None:
            continue
        output_path = vis_dir / "{:06d}_{}.jpg".format(idx, token)
        visualize_sample(info, prediction, dataset, output_path, args)
        saved += 1
        if saved % 10 == 0:
            print("Saved {} visualizations to {}".format(saved, vis_dir))
    print("Saved {} visualizations to {}".format(saved, vis_dir))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info("work dir: {}".format(args.work_dir))

    dataset, data_loader = build_data(cfg, args, distributed)

    if args.local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)

    if args.prediction_pkl is not None:
        predictions = load_pickle(args.prediction_pkl)
    else:
        predictions = run_inference(cfg, args, data_loader, distributed, logger)

    if args.local_rank != 0:
        return

    pred_path = Path(args.work_dir) / "prediction.pkl"
    save_pickle(predictions, pred_path)
    print("Saved predictions to {}".format(pred_path))

    if not args.skip_eval:
        result_dict = evaluate_subset(dataset, predictions, args.work_dir, testset=args.testset)
        if result_dict is not None:
            for key, value in result_dict["results"].items():
                print("Evaluation {}: {}".format(key, value))
            metrics_path = Path(args.work_dir) / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(result_dict, f, indent=2, default=json_default)
            print("Saved metrics to {}".format(metrics_path))

    if args.vis_num != 0:
        visualize_predictions(dataset, predictions, args)


if __name__ == "__main__":
    main()
