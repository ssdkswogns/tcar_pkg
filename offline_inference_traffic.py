#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch import nn
from torchvision import transforms
from ultralytics import YOLO


IDX2LABEL = ["red", "yellow", "green", "greenleft", "redleft"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

PREP = transforms.Compose(
    [
        transforms.Resize((20, 70)),
        transforms.ToTensor(),
    ]
)


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    box_conf: float
    cls_conf: float
    bbox: tuple[int, int, int, int]
    track_length: int = 1
    missed_frames: int = 0


@dataclass
class TrackState:
    bbox: tuple[int, int, int, int]
    cls_id: int
    cls_name: str
    box_conf: float
    cls_conf: float
    hit_streak: int
    last_frame_idx: int
    missed_frames: int = 0
    candidate_cls_id: int | None = None
    candidate_cls_name: str | None = None
    candidate_count: int = 0


class NIA_SEGNet_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = models.mobilenet_v2(weights=None)
        self.fcn.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fcn.last_channel, 5),
        )

    def forward(self, x):
        return self.fcn(x)


def install_lightning_stubs() -> None:
    pl = sys.modules.get("pytorch_lightning")
    if pl is None:
        pl = types.ModuleType("pytorch_lightning")

    callbacks = sys.modules.get("pytorch_lightning.callbacks")
    if callbacks is None:
        callbacks = types.ModuleType("pytorch_lightning.callbacks")

    early = sys.modules.get("pytorch_lightning.callbacks.early_stopping")
    if early is None:
        early = types.ModuleType("pytorch_lightning.callbacks.early_stopping")

    checkpoint = sys.modules.get("pytorch_lightning.callbacks.model_checkpoint")
    if checkpoint is None:
        checkpoint = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class EarlyStopping:  # pragma: no cover - compatibility stub
        pass

    class ModelCheckpoint:  # pragma: no cover - compatibility stub
        pass

    early.EarlyStopping = EarlyStopping
    checkpoint.ModelCheckpoint = ModelCheckpoint
    callbacks.early_stopping = early
    callbacks.model_checkpoint = checkpoint
    pl.callbacks = callbacks

    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules["pytorch_lightning.callbacks.early_stopping"] = early
    sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = checkpoint


def load_checkpoint_compat(ckpt_path: Path):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("pytorch_lightning"):
            raise
        install_lightning_stubs()
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location="cpu")


def load_classifier(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = load_checkpoint_compat(ckpt_path)
    state = ckpt.get("state_dict", ckpt)

    model = NIA_SEGNet_module()
    new_state = {}
    for key, value in state.items():
        if key.startswith("fcn.") or key.startswith("fc."):
            new_state[key] = value
        elif key.startswith("model."):
            new_state[key.replace("model.", "")] = value
        else:
            new_state[key] = value

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load classifier checkpoint cleanly. "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.eval().to(device)
    return model


def xyxy_to_int_bbox(
    xyxy: torch.Tensor,
    width: int,
    height: int,
    pad: int = 2,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy.tolist()
    x1 = max(0, int(np.floor(x1) - pad))
    y1 = max(0, int(np.floor(y1) - pad))
    x2 = min(width - 1, int(np.ceil(x2) + pad))
    y2 = min(height - 1, int(np.ceil(y2) + pad))
    return x1, y1, x2, y2


def draw_label(img_bgr: np.ndarray, text: str, x1: int, y1: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        img_bgr,
        (x1, max(0, y1 - th - 6)),
        (x1 + tw + 4, y1),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img_bgr,
        text,
        (x1 + 2, y1 - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def color_for_label(name: str) -> tuple[int, int, int]:
    if name.startswith("green"):
        return (0, 255, 0)
    if name.startswith("yellow"):
        return (0, 255, 255)
    return (0, 0, 255)


def collect_image_files(input_root: Path, source_dir_name: str) -> list[Path]:
    image_files: list[Path] = []
    folders = sorted(input_root.glob(f"traffic_260323_*/{source_dir_name}"))
    if not folders:
        folders = sorted(p for p in input_root.rglob(source_dir_name) if p.is_dir())

    for folder in folders:
        for image_path in sorted(folder.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                image_files.append(image_path)

    return image_files


def ensure_output_path(image_path: Path, input_root: Path, output_root: Path) -> Path:
    rel_path = image_path.relative_to(input_root)
    output_path = output_root / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def parse_frame_index(image_path: Path) -> int:
    stem_parts = image_path.stem.split("_")
    if not stem_parts or not stem_parts[0].isdigit():
        raise ValueError(f"Could not parse frame index from filename: {image_path.name}")
    return int(stem_parts[0])


def bbox_iou(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def should_relax_thresholds_for_bbox(
    bbox: tuple[int, int, int, int],
    previous_tracks: list[TrackState],
    track_iou_thres: float,
    min_track_frames: int,
) -> bool:
    for track in previous_tracks:
        if track.hit_streak < min_track_frames:
            continue
        if bbox_iou(track.bbox, bbox) >= track_iou_thres:
            return True
    return False


@torch.no_grad()
def run_inference_on_image(
    image_bgr: np.ndarray,
    yolo_model: YOLO,
    cls_model: nn.Module,
    device: torch.device,
    previous_tracks: list[TrackState],
    traffic_class_id: int,
    det_conf_thres: float,
    cls_conf_thres: float,
    min_box_aspect_ratio: float,
    track_iou_thres: float,
    min_track_frames: int,
    tracked_threshold_relaxation: float,
    crop_pad: int,
) -> list[Detection]:
    height, width = image_bgr.shape[:2]
    detections: list[Detection] = []

    result = yolo_model.predict(
        source=image_bgr,
        device=str(device),
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxys = result.boxes.xyxy
    confs = result.boxes.conf
    clses = result.boxes.cls

    for index in range(len(xyxys)):
        yolo_cls = int(clses[index].item())
        if yolo_cls != traffic_class_id:
            continue

        raw_x1, raw_y1, raw_x2, raw_y2 = xyxys[index].tolist()
        box_width = raw_x2 - raw_x1
        box_height = raw_y2 - raw_y1
        if box_width <= 0 or box_height <= 0:
            continue
        # Apply the aspect-ratio filter on the detector box before padding changes it.
        if (box_width / box_height) < min_box_aspect_ratio:
            continue

        x1, y1, x2, y2 = xyxy_to_int_bbox(
            xyxys[index],
            width=width,
            height=height,
            pad=crop_pad,
        )
        relax_thresholds = should_relax_thresholds_for_bbox(
            bbox=(x1, y1, x2, y2),
            previous_tracks=previous_tracks,
            track_iou_thres=track_iou_thres,
            min_track_frames=min_track_frames,
        )

        current_det_conf_thres = det_conf_thres
        current_cls_conf_thres = cls_conf_thres
        if relax_thresholds:
            current_det_conf_thres = max(
                0.0,
                det_conf_thres - tracked_threshold_relaxation,
            )
            current_cls_conf_thres = max(
                0.0,
                cls_conf_thres - tracked_threshold_relaxation,
            )

        box_conf = float(confs[index].item())
        if box_conf < current_det_conf_thres:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = PREP(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
        logits = cls_model(tensor)
        prob = F.softmax(logits, dim=1)[0]
        cls_id = int(torch.argmax(prob).item())
        cls_conf = float(prob[cls_id].item())
        if cls_conf < current_cls_conf_thres:
            continue
        cls_name = IDX2LABEL[cls_id]

        detections.append(
            Detection(
                cls_id=cls_id,
                cls_name=cls_name,
                box_conf=box_conf,
                cls_conf=cls_conf,
                bbox=(x1, y1, x2, y2),
            )
        )

    return detections


def update_tracks(
    detections: list[Detection],
    previous_tracks: list[TrackState],
    frame_idx: int,
    track_iou_thres: float,
    min_track_frames: int,
    track_persist_after_frames: int,
    track_max_missed_frames: int,
    color_change_frames: int,
) -> tuple[list[TrackState], list[Detection]]:
    updated_tracks: list[TrackState] = []
    matched_track_indices: set[int] = set()
    confirmed_detections: list[Detection] = []

    for detection in detections:
        best_track_index = None
        best_iou = 0.0

        for track_index, track in enumerate(previous_tracks):
            if track_index in matched_track_indices:
                continue
            if track.last_frame_idx != frame_idx - 1:
                continue

            overlap = bbox_iou(track.bbox, detection.bbox)
            if overlap < track_iou_thres or overlap <= best_iou:
                continue

            best_iou = overlap
            best_track_index = track_index

        if best_track_index is None:
            updated_track = TrackState(
                bbox=detection.bbox,
                cls_id=detection.cls_id,
                cls_name=detection.cls_name,
                box_conf=detection.box_conf,
                cls_conf=detection.cls_conf,
                hit_streak=1,
                last_frame_idx=frame_idx,
                missed_frames=0,
                candidate_cls_id=None,
                candidate_cls_name=None,
                candidate_count=0,
            )
        else:
            matched_track_indices.add(best_track_index)
            previous_track = previous_tracks[best_track_index]
            stable_cls_id = previous_track.cls_id
            stable_cls_name = previous_track.cls_name
            stable_cls_conf = previous_track.cls_conf
            candidate_cls_id = previous_track.candidate_cls_id
            candidate_cls_name = previous_track.candidate_cls_name
            candidate_count = previous_track.candidate_count

            if detection.cls_id == previous_track.cls_id:
                stable_cls_conf = detection.cls_conf
                candidate_cls_id = None
                candidate_cls_name = None
                candidate_count = 0
            elif previous_track.candidate_cls_id == detection.cls_id:
                candidate_cls_id = detection.cls_id
                candidate_cls_name = detection.cls_name
                candidate_count = previous_track.candidate_count + 1
                if candidate_count >= color_change_frames:
                    stable_cls_id = detection.cls_id
                    stable_cls_name = detection.cls_name
                    stable_cls_conf = detection.cls_conf
                    candidate_cls_id = None
                    candidate_cls_name = None
                    candidate_count = 0
            else:
                candidate_cls_id = detection.cls_id
                candidate_cls_name = detection.cls_name
                candidate_count = 1

            updated_track = TrackState(
                bbox=detection.bbox,
                cls_id=stable_cls_id,
                cls_name=stable_cls_name,
                box_conf=detection.box_conf,
                cls_conf=stable_cls_conf,
                hit_streak=previous_track.hit_streak + 1,
                last_frame_idx=frame_idx,
                missed_frames=0,
                candidate_cls_id=candidate_cls_id,
                candidate_cls_name=candidate_cls_name,
                candidate_count=candidate_count,
            )

        updated_tracks.append(updated_track)
        if updated_track.hit_streak >= min_track_frames:
            confirmed_detections.append(
                Detection(
                    cls_id=updated_track.cls_id,
                    cls_name=updated_track.cls_name,
                    box_conf=updated_track.box_conf,
                    cls_conf=updated_track.cls_conf,
                    bbox=updated_track.bbox,
                    track_length=updated_track.hit_streak,
                    missed_frames=updated_track.missed_frames,
                )
            )

    for track_index, track in enumerate(previous_tracks):
        if track_index in matched_track_indices:
            continue
        if track.hit_streak < track_persist_after_frames:
            continue
        if track.missed_frames >= track_max_missed_frames:
            continue

        carried_track = TrackState(
            bbox=track.bbox,
            cls_id=track.cls_id,
            cls_name=track.cls_name,
            box_conf=track.box_conf,
            cls_conf=track.cls_conf,
            hit_streak=track.hit_streak,
            last_frame_idx=frame_idx,
            missed_frames=track.missed_frames + 1,
            candidate_cls_id=None,
            candidate_cls_name=None,
            candidate_count=0,
        )
        updated_tracks.append(carried_track)
        confirmed_detections.append(
            Detection(
                cls_id=carried_track.cls_id,
                cls_name=carried_track.cls_name,
                box_conf=carried_track.box_conf,
                cls_conf=carried_track.cls_conf,
                bbox=carried_track.bbox,
                track_length=carried_track.hit_streak,
                missed_frames=carried_track.missed_frames,
            )
        )

    return updated_tracks, confirmed_detections


def draw_confirmed_detections(
    image_bgr: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    if not detections:
        return image_bgr

    vis = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_for_label(detection.cls_name), 2)
        draw_label(
            vis,
            (
                f"{detection.cls_name} "
                f"(box={detection.box_conf:.2f}, cls={detection.cls_conf:.2f}, "
                f"trk={detection.track_length}, miss={detection.missed_frames})"
            ),
            x1,
            y1,
        )
    return vis


def infer_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but this Python environment does not have CUDA-enabled PyTorch."
        )
    return device


def print_progress(
    index: int,
    total: int,
    image_path: Path,
    detections: Iterable[Detection],
) -> None:
    names = [det.cls_name for det in detections]
    summary = ", ".join(names) if names else "none"
    print(f"[{index}/{total}] {image_path.name} -> detections: {summary}")


def parse_args() -> argparse.Namespace:
    package_dir = Path(__file__).resolve().parents[1]
    model_dir = package_dir / "models"

    parser = argparse.ArgumentParser(
        description="Offline traffic light inference with saved visualization outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/shchon11/traffic_260323/extracted_images"),
        help="Root directory containing traffic_260323_* folders.",
    )
    parser.add_argument(
        "--source-dir-name",
        type=str,
        default="camera_3__compressed",
        help="Image folder name to search under each sequence directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory where visualization images will be written. Defaults to <input-root>_inference_vis.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=model_dir / "last.pt",
        help="Path to YOLO weights.",
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=Path,
        default=model_dir / "mobilenet.ckpt",
        help="Path to classifier checkpoint.",
    )
    parser.add_argument(
        "--traffic-class-id",
        type=int,
        default=1,
        help="YOLO class id for traffic_light.",
    )
    parser.add_argument(
        "--det-conf-thres",
        type=float,
        default=0.6,
        help="Minimum YOLO box confidence to keep.",
    )
    parser.add_argument(
        "--cls-conf-thres",
        type=float,
        default=0.9,
        help="Minimum classifier confidence to keep.",
    )
    parser.add_argument(
        "--min-box-aspect-ratio",
        type=float,
        default=2.0,
        help="Minimum detector bbox width/height ratio to keep.",
    )
    parser.add_argument(
        "--track-iou-thres",
        type=float,
        default=0.1,
        help="IoU threshold for treating detections in adjacent frames as the same object.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=3,
        help="Minimum consecutive matched frames required before a detection is considered final.",
    )
    parser.add_argument(
        "--track-persist-after-frames",
        type=int,
        default=10,
        help="Tracks with at least this many matched frames can survive temporary missed detections.",
    )
    parser.add_argument(
        "--track-max-missed-frames",
        type=int,
        default=5,
        help="Maximum missed frames allowed for mature tracks before dropping them.",
    )
    parser.add_argument(
        "--tracked-threshold-relaxation",
        type=float,
        default=0.2,
        help="How much to lower detector/classifier thresholds for bbox-overlapping confirmed tracks.",
    )
    parser.add_argument(
        "--color-change-frames",
        type=int,
        default=4,
        help="Number of consecutive frames with a new color before a tracked object changes its stable color.",
    )
    parser.add_argument(
        "--crop-pad",
        type=int,
        default=2,
        help="Padding pixels added around the detector box before classification.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device. Use auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap for quick testing. 0 means process all images.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose visualization output already exists.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print progress every N processed images.",
    )
    parser.add_argument(
        "--save-after-last-bbox",
        type=int,
        default=50,
        help="Save frames only while a bbox has appeared within the last N frames. 0 saves detected frames only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = infer_device(args.device)

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if not args.yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    if not args.classifier_ckpt.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {args.classifier_ckpt}")
    if args.output_root is None:
        args.output_root = args.input_root.parent / f"{args.input_root.name}_inference_vis5"

    image_files = collect_image_files(args.input_root, args.source_dir_name)
    if not image_files:
        raise FileNotFoundError(
            f"No images found under {args.input_root} with folder name {args.source_dir_name}"
        )

    if args.max_images > 0:
        image_files = image_files[: args.max_images]

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Images to process: {len(image_files)}")
    print(f"YOLO weights: {args.yolo_weights}")
    print(f"Classifier checkpoint: {args.classifier_ckpt}")
    print(f"Output root: {args.output_root}")
    print(f"Detector conf threshold: {args.det_conf_thres}")
    print(f"Classifier conf threshold: {args.cls_conf_thres}")
    print(f"Min bbox aspect ratio: {args.min_box_aspect_ratio}")
    print(f"Track IoU threshold: {args.track_iou_thres}")
    print(f"Min track frames: {args.min_track_frames}")
    print(f"Track persist after frames: {args.track_persist_after_frames}")
    print(f"Track max missed frames: {args.track_max_missed_frames}")
    print(f"Tracked threshold relaxation: {args.tracked_threshold_relaxation}")
    print(f"Color change frames: {args.color_change_frames}")
    print(f"Save window after last bbox: {args.save_after_last_bbox}")

    yolo_model = YOLO(str(args.yolo_weights))
    cls_model = load_classifier(args.classifier_ckpt, device)

    saved_count = 0
    skipped_count = 0
    skipped_by_policy_count = 0
    detected_count = 0
    current_sequence_dir: Path | None = None
    frames_since_last_detection: int | None = None
    active_tracks: list[TrackState] = []

    for index, image_path in enumerate(image_files, start=1):
        if image_path.parent != current_sequence_dir:
            current_sequence_dir = image_path.parent
            frames_since_last_detection = None
            active_tracks = []

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        frame_idx = parse_frame_index(image_path)
        raw_detections = run_inference_on_image(
            image_bgr=image_bgr,
            yolo_model=yolo_model,
            cls_model=cls_model,
            device=device,
            previous_tracks=active_tracks,
            traffic_class_id=args.traffic_class_id,
            det_conf_thres=args.det_conf_thres,
            cls_conf_thres=args.cls_conf_thres,
            min_box_aspect_ratio=args.min_box_aspect_ratio,
            track_iou_thres=args.track_iou_thres,
            min_track_frames=args.min_track_frames,
            tracked_threshold_relaxation=args.tracked_threshold_relaxation,
            crop_pad=args.crop_pad,
        )
        active_tracks, detections = update_tracks(
            detections=raw_detections,
            previous_tracks=active_tracks,
            frame_idx=frame_idx,
            track_iou_thres=args.track_iou_thres,
            min_track_frames=args.min_track_frames,
            track_persist_after_frames=args.track_persist_after_frames,
            track_max_missed_frames=args.track_max_missed_frames,
            color_change_frames=args.color_change_frames,
        )
        vis = draw_confirmed_detections(image_bgr, detections)

        if detections:
            detected_count += 1
            frames_since_last_detection = 0
            should_save = True
        else:
            if frames_since_last_detection is None:
                should_save = False
            else:
                frames_since_last_detection += 1
                should_save = frames_since_last_detection <= args.save_after_last_bbox

        if should_save:
            output_path = ensure_output_path(image_path, args.input_root, args.output_root)
            if args.skip_existing and output_path.exists():
                skipped_count += 1
            elif not cv2.imwrite(str(output_path), vis):
                print(f"[WARN] Failed to write output: {output_path}")
            else:
                saved_count += 1
        else:
            skipped_by_policy_count += 1

        should_print = (
            index == 1
            or index == len(image_files)
            or (args.print_every > 0 and index % args.print_every == 0)
        )
        if should_print:
            print_progress(index, len(image_files), image_path, detections)

    print("Done.")
    print(f"Saved visualizations: {saved_count}")
    print(f"Skipped existing outputs: {skipped_count}")
    print(f"Skipped by save policy: {skipped_by_policy_count}")
    print(f"Images with detections: {detected_count}")


if __name__ == "__main__":
    main()
